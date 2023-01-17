# -*- coding: utf-8 -*-

"""

Created on Thu Jun 20 23:03:37 2019



@author: AnubhavA

"""



from bs4 import BeautifulSoup

import urllib.request

import pandas as pd

from pandas import ExcelWriter

##for displaying dataframe in notebooks

from IPython.display import display, HTML

#for progress showcase

#from tqdm import tqdm

#from tqdm import tqdm_notebook as tqdm

import base64



def save_xls(list_dfs, xls_path):

    "function to save dataframe data to excel sheet file"

    writer = ExcelWriter(xls_path)

    for n, df in enumerate(list_dfs):

        df.to_excel(writer,'sheet%s' % n,index=False)

    writer.save()



    

user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'

headers={'User-Agent':user_agent,} 

        

 
##---------function to get url data from career-list pages

def generate_url_data_sheet():

    url_columns = ['menu_name', 'leaf_name', 'url_overview', 'url_skills']

    df_url = pd.DataFrame(columns=url_columns)

    

    url = "https://www.mymajors.com/career-list/"

    

    request=urllib.request.Request(url,None,headers) #The assembled request

    response = urllib.request.urlopen(request)

    data = response.read()

    soup = BeautifulSoup(data, 'lxml')   

    

    for ul in soup.find_all('ul', {'class' : 'menu'}):

        name_head_li = ""

        for li in ul.find_all('li', {'class' : 'expanded top'}):

            name_head_li = li.find('a').contents[0]

            for ul_menu in li.find_all('ul', {'class' : 'menu'}):

                for li_leaf in ul_menu.find_all('li', {'class' : 'leaf'}):

                    href_leaf = li_leaf.find('a')['href']

                    ##there are some urls missing end / 

                    if("All-Other" in href_leaf):

                        href_leaf = href_leaf + "/"                    

                    href_leaf = href_leaf.replace('..', 'https://www.mymajors.com', 1)

                    href_leaf_skills = href_leaf + "skills"

                    name_leaf = li_leaf.find('a').contents[0]

                    df_url = df_url.append({'menu_name': name_head_li, 'leaf_name': name_leaf, 'url_overview':href_leaf, 'url_skills': href_leaf_skills}, ignore_index=True)  

 



    return df_url
##---------function to get data from overview pages

def generate_overview_data_sheet(df_url):

    count = 0

    data_columns_overview = ['menu_name', 'leaf_name', 'url_overview', 'importance', 'activities']

    df_data_overview = pd.DataFrame(columns=data_columns_overview)

    

    for index, row in df_url.iterrows():

        if count < df_url.shape[0]:

            print("overview page last processed : {0}".format(count), end='\r')

            count = count + 1

            menu_name = row['menu_name'] 

            leaf_name = row['leaf_name'] 

            url_overview = row['url_overview']

            

            request=urllib.request.Request(url_overview,None,headers) #The assembled request

            response = urllib.request.urlopen(request)

            data = response.read()

            soup = BeautifulSoup(data, 'lxml')

                    

            tables = soup.findChildren('table')

            for table in tables:

                rows = table.findChildren(['th', 'tr'])             

                for row in rows:

                    cells = row.findChildren('td')

                    if (len(cells)>0):

                        cell1 = cells[0]

                        cell1_img_1_width = cell1.find('img')['width']

                        

                        cell2 = cells[1]

                        cell2_value = cell2.string

                        df_data_overview = df_data_overview.append({'menu_name': menu_name, 'leaf_name': leaf_name, 'url_overview':url_overview, 'importance': cell1_img_1_width, 'activities': cell2_value}, ignore_index=True)

                    

    return df_data_overview         
# ----------function to get data from skill page for three tables skills, knowledge, styles

def generate_skill_page_data_sheet(df_url):

    data_columns_skill = ['menu_name', 'leaf_name', 'url_skills', 'importance', 'skills']

    df_data_skill = pd.DataFrame(columns=data_columns_skill)

    

    data_columns_knowledge = ['menu_name', 'leaf_name', 'url_skills', 'importance', 'knowledge']

    df_data_knowledge = pd.DataFrame(columns=data_columns_knowledge)

    

    data_columns_styles = ['menu_name', 'leaf_name', 'url_skills', 'importance', 'styles']

    df_data_styles = pd.DataFrame(columns=data_columns_styles)

    count = 0

    for index, row in df_url.iterrows():

        if count < df_url.shape[0]:            

            print("processing skill page : {0}".format(count), end='\r')

            count = count + 1

            menu_name = row['menu_name'] 

            leaf_name = row['leaf_name'] 

            url_skills = row['url_skills']

        

            request=urllib.request.Request(url_skills,None,headers) #The assembled request

            response = urllib.request.urlopen(request)

            data = response.read()

            soup = BeautifulSoup(data, 'lxml')

                    

            tables = soup.findChildren('table')

            for index , table in enumerate(tables):

#                 print(index)

                if index==0:

                    ## fetching data for table skills

                    rows = table.findChildren(['th', 'tr'])             

                    for row in rows:

                        cells = row.findChildren('td')

                        if (len(cells)>0):

                            cell1 = cells[0]

                            cell1_img_1_width = cell1.find('img')['width']

                            

                            cell2 = cells[1]

                            cell2_value = cell2.string

                            df_data_skill = df_data_skill.append({'menu_name': menu_name, 'leaf_name': leaf_name, 'url_skills':url_skills, 'importance': cell1_img_1_width, 'skills': cell2_value}, ignore_index=True)

                    

                if index==1:

                     ## fetching data for table knowledge

                    rows = table.findChildren(['th', 'tr'])             

                    for row in rows:

                        cells = row.findChildren('td')

                        if (len(cells)>0):

                            cell1 = cells[0]

                            cell1_img_1_width = cell1.find('img')['width']



                            cell2 = cells[1]

                            cell2_value = cell2.string

                            df_data_knowledge = df_data_knowledge.append({'menu_name': menu_name, 'leaf_name': leaf_name, 'url_skills':url_skills, 'importance': cell1_img_1_width, 'knowledge': cell2_value}, ignore_index=True)



                if index==2:

                    ## fetching data for table styles

                    rows = table.findChildren(['th', 'tr'])             

                    for row in rows:

                        cells = row.findChildren('td')

                        if (len(cells)>0):

                            cell1 = cells[0]

                            cell1_img_1_width = cell1.find('img')['width']



                            cell2 = cells[1]

                            cell2_value = cell2.string

                            df_data_styles = df_data_styles.append({'menu_name': menu_name, 'leaf_name': leaf_name, 'url_skills':url_skills, 'importance': cell1_img_1_width, 'styles': cell2_value}, ignore_index=True)



    return df_data_skill, df_data_knowledge, df_data_styles

# --- fetchin and saving url data to excelsheet 

df_url = generate_url_data_sheet()

print("Count of URLS found: {0}".format(df_url.shape[0])) ##1051    



print("Sample url data:")



display(df_url.head(2))

sheets =[df_url] 



save_xls(sheets,'urls.xlsx')

print("data saved as urls.xlsx")
# ---UNCOMMENT to fetch and save  Overview data to excelsheet 



# df_data_overview = generate_overview_data_sheet(df_url) 

# print("Sample Overview data:")



# display(df_data_overview.head(2))

# sheets =[df_data_overview] 



# save_xls(sheets,'Overview.xlsx') 

# print("Overview data saved as Overview.xlsx")
#df_url.loc[239][3]
 # --- fetchin and saving  skillpages data to excelsheet

##-- table 1 

df_data_skill, df_data_knowledge, df_data_styles = generate_skill_page_data_sheet(df_url)

print("Sample skills table data:")

display(df_data_skill.head(2))

sheets =[df_data_skill] 

save_xls(sheets,'skills.xlsx')   



##-- table 2

print("Sample knowledge table data:")

display(df_data_knowledge.head(2))

sheets =[df_data_knowledge] 

save_xls(sheets,'knowledge.xlsx')  



##-- table 3

print("Sample styles table data:")

display(df_data_styles.head(2))

sheets =[df_data_styles] 

save_xls(sheets,'styles.xlsx') 



print("skills, knowledge, styles tables data saved as skills.xlsx, knowledge.xlsx and styles.xlsx respectively ")

    