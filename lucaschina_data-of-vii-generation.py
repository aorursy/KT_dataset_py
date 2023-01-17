import os

from selenium import webdriver

#from selenium.webdriver.common.by import By

from webdriver_manager.chrome import ChromeDriverManager

from time import sleep

import pandas as pd



driver = webdriver.Chrome(ChromeDriverManager().install())



#shape dict, collected by hand

d = {'Shape 06':'bipedal_tailed',

     'Shape 13':'four_wings',

     'Shape 04':'head_arms',

     'Shape 05':'head_base',

     'Shape 07':'head_legs',

     'Shape 01':'head_only',

     'Shape 14':'insectoid',

     'Shape 11':'multiple_bodies',

     'Shape 08':'quadruped',

     'Shape 02':'serpentine_body',

     'Shape 10':'several_limbs',

     'Shape 09':'two_wings',

     'Shape 03':'with_fins',

     'Shape 12':'human'}



#grab function return a list containing information I need, you can add more.

#the way to find xpath is firstly press 'F12' on the website, suggest you to use chrome. Then go to the exact place you want - right click - choose copy the xpath.

def grab(s):

    #input your pokemon

    web_address='https://pokemon.fandom.com/wiki/'+s  #find the website

    

    driver.get(web_address) #open the website

    

    #this is to get the first type(might have two) of the pokemon

    try:

        type1 = driver.find_element_by_xpath("//*[@id='mw-content-text']/aside/section[2]/div[1]/div/a[1]")

        type1 = type1.get_attribute('title')

        type1 = type1.split(' ')[0]

    except:

        type1 = ''

    #to get the first egg_group

    try:

        egg_group1 = driver.find_element_by_xpath("//*[@id='mw-content-text']/aside/section[6]/section[3]/section[2]/div[1]/a")

        egg_group1 = egg_group1.text

    except:

        egg_group1 = ''

    #to get the color

    try:

        color = driver.find_element_by_xpath("//*[@id='mw-content-text']/aside/section[6]/section[2]/section[2]/div[2]/font")

        color = color.text

    except:

        color = ''

    #to get the shape

    try:

        shape = driver.find_element_by_xpath("//*[@id='mw-content-text']/aside/section[6]/section[3]/section[2]/div[2]/div/div/a")

        shape = d[shape.get_attribute('title')]

    except:

        shape = ''    

    return [type1, egg_group1, color, shape]
num = []

name = []

type1 = []

egg_group1 = []

color = []

shape = []

fail = []



n = len(pokemon_list)

for i in range(n):

    x = pokemon_list[i]

    try:

        res = grab(x)

        num.append(i+1)

        name.append(x)

        type1.append(res[0])

        egg_group1.append(res[1])

        color.append(res[2])

        shape.append(res[3])

        #print(i)

    except:

        res = ['','','','']

        type1.append(res[0])

        egg_group1.append(res[1])

        color.append(res[2])

        shape.append(res[3])

        fail.append(x)

        #print(x)

        sleep(1)



df = pd.DataFrame({'num':num,'name':name,'type':type1,'egg_group':egg_group1,'color':color,'shape':shape})

os.chdir(r'C:\Users\rsslu\Desktop\DASC\tag')



df.to_csv('7_gen_tag.csv')