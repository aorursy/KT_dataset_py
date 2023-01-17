from matplotlib import pyplot as plt

import matplotlib.animation as animation

import random

import numpy as np

import time

from tkinter import *

from pylab import *

from drawnow import drawnow

import random

import tkinter as tk

import PySimpleGUI as sg

import os

import pyautogui

import sys

from PIL import ImageGrab

%matplotlib inline


def GetInputs():

    

    sg.theme('Topanga')      # Add some color to the window



    layout = [

    

        [sg.Text('Please Enter Data for Covid 19 Simulation', size=(30,2), justification='center', font=("Helvetica", 30), auto_size_text = True, relief=sg.RELIEF_RIDGE)],

        [sg.Text()],

        [sg.Text('Population Sample size', size=(30, 1)), sg.Slider(range=(1, 1000), orientation='h', size=(35, 20), default_value=500)],   

        [sg.Text('No. of Infected people', size=(30, 1)), sg.InputText(15)],

        [sg.Text()],

        [sg.Text('% Elderly people', size=(30, 1)), sg.Slider(range=(1, 100), orientation='h', size=(35, 20), default_value=20)],   

        [sg.Text('% of Elderly in Home Care', size=(30, 1)), sg.Slider(range=(1, 100), orientation='h', size=(35, 20), default_value=5)],   

        [sg.Frame(layout=[      

        [sg.Checkbox('Testing + Quarantine?', default=True, size=(20,1)),  sg.Checkbox('PPE on?', default=True, size=(20,1)), sg.Checkbox('Social Distancing on?', default=True, size=(20,1)), sg.Checkbox('Live Daily Output??', default = True, size=(20,1))],      

        [sg.Radio('Start with Random model     ', "RADIO1", default=True, size=(30,1)), sg.Radio('Reproduce Results', "RADIO1")]], title='Options',title_color='red', relief=sg.RELIEF_SUNKEN, tooltip='Use these to set flags')],      

        [sg.Text('Probability of Young to reco from Hos.', size=(30, 1)), sg.Slider(range=(1, 100), orientation='h', size=(35, 20), default_value=90)],   

        [sg.Text('Probability of Old  to reco from Hos.', size=(30, 1)), sg.Slider(range=(1, 100), orientation='h', size=(35, 20), default_value=80)],

        [sg.Text('Prob.  of patient death in ICU (%)', size=(30, 1)), sg.InputText(1)],   

        [sg.Text('Reference Case Study, Insert Title:', size=(30, 1)), sg.InputText('Case 001')],

        [sg.Checkbox('Save a Screenshot of Inputs?', default=False, size=(30,1))],      

        [sg.Button('Submit'), sg.Button('Cancel')]

        ]

    

    windowmain = sg.Window('Covid-19 data entry window', layout, location =(710,0))

    event, values = windowmain.read()

    

    if event == 'Cancel': os._exit(0)

    valuesmain = values

    

    if valuesmain[4] == False and valuesmain[5] == False and valuesmain[6] == False: windowmain.close()

    

    # Testing Option is on

    if valuesmain[4] == True:

        sg.ChangeLookAndFeel('GreenTan')      

    

        layout = [

            [sg.Text('TESTING & QUARANTINE OPTION')],     

            [sg.Text('% of Population to be tested', size=(30, 1)), sg.Slider(range=(1, 100), orientation='h', size=(35, 20), default_value=10)],   

            [sg.Text('Day Testing to START after 1st infection', size=(30, 1)), sg.InputText(1)],  

            [sg.Text('Day Testing to END after 1st infection', size=(30, 1)), sg.InputText(200)],  

            [sg.Button('Submit'), sg.Button('Cancel')]

        ]

        windowtest = sg.Window('Testing & Quarantine Window', layout, default_element_size=(40, 1), location =(0,0),grab_anywhere=False)      

        event, values = windowtest.read()  

        

        if event == 'Cancel': os._exit(0)

        valuestest = values



      # Personal Protection Equipmemt Option is on

    if valuesmain[5] == True:

        sg.ChangeLookAndFeel('GreenTan')      

    

        layout = [

            [sg.Text('PERSONAL PROTECTION EQUIPMEMT OPTION')],     

            [sg.Text('% of Population wearing PPE', size=(30, 1)), sg.Slider(range=(1, 100), orientation='h', size=(35, 20), default_value=10)],   

            [sg.Text('Day PPE to START after 1st infection', size=(30, 1)), sg.InputText(1)],  

            [sg.Text('Day PPE to END after 1st infection', size=(30, 1)), sg.InputText(200)],  

            [sg.Text('% Protection from PPE', size=(30, 1)), sg.Slider(range=(1, 100), orientation='h', size=(35, 20), default_value=90)],   



            [sg.Button('Submit'), sg.Button('Cancel')]

        ]

        windowppe = sg.Window('Personal Protection Equipmemt Window', layout, default_element_size=(40, 1), location =(0,220),grab_anywhere=False)      

        event, values = windowppe.read()   

        

        if event == 'Cancel': os._exit(0)

        valuesppe = values

    

# Social Distancing Option is on

    if valuesmain[6] == True:

        sg.ChangeLookAndFeel('GreenTan')      

    

        layout = [

            [sg.Text('SOCIAL DISTANCING OPTION')],     

            [sg.Text('% of Population practising SD', size=(30, 1)), sg.Slider(range=(1, 100), orientation='h', size=(35, 20), default_value=10)],   

            [sg.Text('Day SD to START after 1st infection', size=(30, 1)), sg.InputText(1)],  

            [sg.Text('Day SD to END after 1st infection', size=(30, 1)), sg.InputText(200)],  

            [sg.Text('% Protection from SD', size=(30, 1)), sg.Slider(range=(1, 100), orientation='h', size=(35, 20), default_value=90)],   

    

            [sg.Button('Submit'), sg.Button('Cancel')]

        ]

        windowsd = sg.Window('Social Distancing Window', layout, default_element_size=(40, 1), location =(0,480),grab_anywhere=False)      

        event, values = windowsd.read()   

        

        if event == 'Cancel': os._exit(0)

        valuessd = values



# Read in File name for Screenshot saving

    if valuesmain[14] == True:



        if len(sys.argv) == 1:

            event, values = sg.Window('ScreenShot of Inputs',

                            [[sg.Text('Document to save Input')],

                            [sg.In(), sg.FileBrowse()],

                            [sg.Open(), sg.Cancel()]]).read(close=True)

            fname = values[0]

        else:

            fname = sys.argv[1]



        if not fname:

            sg.popup("Cancel", "No filename supplied")

            raise SystemExit("Cancelling: no filename supplied")

        else:

            sg.popup('The filename you chose was', fname)



        image = ImageGrab.grab(bbox=(0,0,1460,770))

        image.save(fname)

        event = "Submit"

        

    if valuesmain[4] == True: windowtest.close()

    if valuesmain[5] == True: windowppe.close()

    if valuesmain[6] == True: windowsd.close()

#    if valuesmain[13] == True: windowsshot.close()

    windowmain.close()



    # Population Sample size

    npeople = int(valuesmain[0])

    # Number of people initially infected (on day zero)

    initialinfect = int(valuesmain[1])

    ninfected = initialinfect

    

    # % of old people in Population Sample

    percentold = valuesmain[2]

    # % of old people in Care Home

    percentcarehome = valuesmain[3]

    

    #testing on/off

    testing = valuesmain[4]

    #PPE on/off

    ppe= valuesmain[5]

    #Social Distance on/off

    socdis= valuesmain[6]

    #Live daily Result on/off

    livedaily = valuesmain[7]

    

    #Start with random distribution on/off

    if valuesmain[8]== True: FREEZE = False

    if valuesmain[8] == False: FREEZE = True

    

    #Probability of Young to recover from Quarantine/Hospital

    percentyreco = valuesmain[10]

    # Probability (%) of old to Recover from Hospital

    percentoreco = valuesmain[11]

    #probability of patient death in ICU

    percentdicu = float(valuesmain[12])

#    print("valuesmain   = ", valuesmain)

    reference = valuesmain[13]



    # % Covid Testing of Population Sample

    if testing == True:

        percentcovtest = valuestest[0]

        daystarttest = int(valuestest[1])

        dayendtest = int(valuestest[2])

    else:

        percentcovtest = 0

        daystarttest = 0

        dayendtest = 0



    

    # % wearing PPE

    if ppe == True:

        percentppe = valuesppe[0]

        daystartppe = int(valuesppe[1])

        dayendppe = int(valuesppe[2])

        # % Efficiency of PPE

        percenteppe= valuesppe[3]

    else:

        percentppe = 0

        daystartppe = 0

        dayendppe = 0

        # % Efficiency of PPE

        percenteppe= 0



    

    # % of Population Practising Social Distancing

    if socdis == True:

        percentsd = valuessd[0]

        daystartsd = int(valuessd[1])

        dayendsd = int(valuessd[2])

    else:

        percentsd = 0

        daystartsd = 0

        dayendsd = 0



    

    if npeople < 101:

        WIDTH = 1000

        HEIGHT =800

        peoplesizex = 20

        peoplesizey = 20

        peoplespeedx = 10

        peoplespeedy = 10

        contactdist = peoplesizex * 1.25

    else:

        WIDTH = 1700

        HEIGHT = 850

        peoplesizex = 10

        peoplesizey = 10

        peoplespeedx = 10

        peoplespeedy = 10

        contactdist = peoplesizex * 1.25

    

        

        

    nold = int((percentold/100) * npeople)

    ncarehome = int((percentcarehome/100) * nold)

    #Note new Definition of pep!!!!!!!!!!!!!!!!!!!!!!!!

    nppe = int((percentppe/100) * npeople)

    npsd = int((percentsd/100) * npeople)

    

    #Probability/Efficiency  of ppe

    probppe = percenteppe/100.0

    #Probability/Efficiency  of Social Distance

    probsd = percentsd/100.0

    #Probability/Efficiency  of Cov testing

    probcovt = percentcovtest/100.0

    

    #Probability of Young Recover

    probyreco = percentyreco/100.0

    #Probability/ of Old recover

    proboreco= percentoreco/100.0

    #Probability of patient death in ICU

    probdicu = percentdicu/100.0

 

#Set startrun to True or false

    startrun = False

    if event == 'Submit':

        startrun = True



    covidlogicinput = []

    covidlogicinput.insert(0, testing)

    covidlogicinput.insert(1, ppe)

    covidlogicinput.insert(2, socdis)

    covidlogicinput.insert(3, livedaily)

    covidlogicinput.insert(4, FREEZE)

    covidlogicinput.insert(5, startrun)

    covidlogicinput.insert(6,reference)



    

    covidpeopleinput = []

    covidpeopleinput.insert(0, npeople)

    covidpeopleinput.insert(1, ninfected)

    covidpeopleinput.insert(2, nold)

    covidpeopleinput.insert(3, ncarehome)



    covidpeopleinput.insert(4, nppe)

    covidpeopleinput.insert(5, probppe)

    covidpeopleinput.insert(6, npsd)

    covidpeopleinput.insert(7, probsd)

    covidpeopleinput.insert(8, probcovt)

    covidpeopleinput.insert(9, probyreco)

    covidpeopleinput.insert(10, proboreco)

    covidpeopleinput.insert(11, probdicu)



    

    coviddimension = []

    coviddimension.insert(0, WIDTH)

    coviddimension.insert(1, HEIGHT)

    coviddimension.insert(2, peoplesizex)

    coviddimension.insert(3, peoplesizey)

    coviddimension.insert(4, peoplespeedx)

    coviddimension.insert(5, peoplespeedy)

    coviddimension.insert(6, contactdist)

    coviddimension.insert(7, startrun)

    

    coviddays = []

    coviddays.insert(0,daystarttest)

    coviddays.insert(1,dayendtest)

    coviddays.insert(2,daystartppe)

    coviddays.insert(3,dayendppe)

    coviddays.insert(4,daystartsd)

    coviddays.insert(5,dayendsd)



    return covidpeopleinput, coviddimension, coviddays, covidlogicinput



#Get Inputs

covidpeopleinput, coviddimension, coviddays, covidlogicinput = GetInputs()
livedaily = covidlogicinput[3]



FREEZE= covidlogicinput[4]

if FREEZE== True:

    random.seed(301)

    np.random.seed(301)

    

WIDTH = coviddimension[0]

HEIGHT = coviddimension[1]



npeople = covidpeopleinput[0]



initialinfect = covidpeopleinput[1]

ninfected = initialinfect



nold = covidpeopleinput[2]

ncarehome = covidpeopleinput[3]

npep = covidpeopleinput[4]

npsd = covidpeopleinput[6]



peoplesizex = coviddimension[2]

peoplesizey = coviddimension[3]

peoplespeedx = coviddimension[4]

peoplespeedy = coviddimension[5]

contactdist = coviddimension[6]



QWIDTH = int(WIDTH / 4)

QHEIGHT = int(HEIGHT / 3)

XBUF = 20

YBUF= 20



#Set for Countour Plot

ngridx = 5

ngridy = 5



QEXBUF = WIDTH - QWIDTH -XBUF

QILXBUF = QEXBUF + 2*XBUF

QIRXBUF = WIDTH - XBUF

QEYBUF = QHEIGHT + YBUF

QIBYBUF = QEYBUF - 2*YBUF

QITYBUF = YBUF



ICUWIDTH = QWIDTH

ICUHEIGHT = QHEIGHT





ICUEXBUF = WIDTH - ICUWIDTH -XBUF

ICUILXBUF = ICUEXBUF + 2*XBUF

ICUIRXBUF = WIDTH - XBUF

ICUEYBUF = HEIGHT - ICUHEIGHT-YBUF

ICUITYBUF = ICUEYBUF + 2*YBUF

ICUIBYBUF = HEIGHT - YBUF
tk = Tk()

canvas = Canvas(tk, width=WIDTH, height=HEIGHT, bg="black")

tk.title("Covid Propagation Simulation")

canvas.create_rectangle((WIDTH-QWIDTH), 0, WIDTH, QHEIGHT, fill ="yellow")

lbl=Label(tk, text="14 Days Quarantine/Hospital", fg='blue', font=("Helvetica", 10))



lbl.place(x=(WIDTH-QWIDTH+XBUF), y=5)

lbl2=Label(tk, text="ICU (White=Young; Black=Old; Brown=Care Home;Purple=Deceased", fg='blue', font=("Helvetica", 8))

lbl2.place(x=(WIDTH-ICUWIDTH+5), y=ICUITYBUF-10)

lbl3=Label(tk, text="Green = Non-Infected; Red = Infected; Yellow = Recovered", fg='blue', font=("Helvetica", 10))

lbl3.place(x=XBUF, y=HEIGHT-20)





canvas.create_rectangle((WIDTH-ICUWIDTH), (HEIGHT-ICUHEIGHT), WIDTH, HEIGHT, fill ="red")



canvas.pack()

def daylabel(jday):

    textday = 'Day ' + str(jday)

    lbl=Label(tk, text=textday, fg='black', font=("Helvetica", 20))

    lbl.place(x=5, y=5)
class Person:

    def __init__(self, colour, xsize, ysize, xspeed, yspeed):

        

        self.xstart = random.randrange(0, WIDTH-xsize)

        self.ystart = random.randrange(0, HEIGHT-ysize)

        self.xend   = self.xstart + xsize

        self.yend   = self.ystart + ysize

        

        if self.xend >= QEXBUF and self.ystart <= QEYBUF:

            self.xend = 2.0 * QEXBUF - self.xend

            self.xstart = self.xend - xsize

            

        if self.xend >= ICUEXBUF and self.ystart >= ICUEYBUF:

            self.xend = 2.0 * ICUEXBUF - self.xend

            self.xstart = self.xend - xsize

            

        self.colour = colour

        self.xspeed = random.randrange(-xspeed, xspeed)

        self.yspeed = random.randrange(-yspeed, yspeed)

                          

        self.shape = canvas.create_oval(self.xstart, self.ystart, self.xend, self.yend, fill=self.colour)

        self.xcentre = (self.xend + self.xstart)/2

        self.ycentre = (self.yend + self.ystart)/2

    

    def move(self):

        canvas.move(self.shape, self.xspeed, self.yspeed)

        self.newpos = canvas.coords(self.shape)

        self.xcentre = (self.newpos[0] + self.newpos[2])/2

        self.ycentre = (self.newpos[1] + self.newpos[3])/2

        

        if self.colour == "green" or self.colour =="red" or self.colour =="yellow":

         

            if self.newpos[3] >= HEIGHT or self.newpos[1] <=0:

                self.yspeed = -self.yspeed

            if self.newpos[2] >= WIDTH or self.newpos[0] <=0:

                self.xspeed = - self.xspeed

            

            if self.newpos[2] >= QEXBUF and self.newpos[1] <=QEYBUF:

                self.yspeed = -self.yspeed

                self.xspeed = - self.xspeed

            

            if self.newpos[2] >= ICUEXBUF and self.newpos[3] >=ICUEYBUF:

                self.yspeed = -self.yspeed

                self.xspeed = - self.xspeed

            

        if self.colour == "blue":

            if self.newpos[3] >= QIBYBUF or self.newpos[1] <= QITYBUF:

                self.yspeed = -self.yspeed

            if self.newpos[2] >= QIRXBUF or self.newpos[0] <= QILXBUF:

                self.xspeed = -self.xspeed

                

        if self.colour == "white" or self.colour == "black":

            if self.newpos[3] >= ICUIBYBUF or self.newpos[1] <= ICUITYBUF:

                self.yspeed = -self.yspeed

            if self.newpos[2] >= ICUIRXBUF or self.newpos[0] <= ICUILXBUF:

                self.xspeed = -self.xspeed



         

    def changecolour(self, newcol):

        self.colour = newcol

        canvas.itemconfig(self.shape, fill= newcol) # change color

        

    def movequa(self, newpos):

        qx = random.randrange((WIDTH - QWIDTH+XBUF), WIDTH-XBUF)

        qy = random.randrange(XBUF, QHEIGHT-XBUF)

        qxspeed = qx - newpos[0]

        qyspeed = qy - newpos[1]

        canvas.move(self.shape, qxspeed, qyspeed)

        

    def moveicu(self, newpos):

        qx = random.randrange((WIDTH - ICUWIDTH+XBUF), WIDTH-XBUF)

        qy = random.randrange((HEIGHT-ICUHEIGHT)+XBUF, HEIGHT-XBUF)

        qxspeed = qx - newpos[0]

        qyspeed = qy - newpos[1]

        canvas.move(self.shape, qxspeed, qyspeed)

        

    def movebac(self, newpos):

        qx = random.randrange(XBUF, (WIDTH - QWIDTH-XBUF))

        qy = random.randrange(XBUF, HEIGHT-XBUF)

        qxspeed = qx - newpos[0]

        qyspeed = qy - newpos[1]

        canvas.move(self.shape, qxspeed, qyspeed)
def peopleage(people, npeople, nold, ncarehome):

    oldlist = random.sample(range(0, (npeople-1)), nold)

    for j in range(npeople):

        people[j].age = 'young'

        

    for i in range(nold):

        j = oldlist[i]

        if i < ncarehome:

            people[j].age = 'carehome'

            people[j].xspeed = 0

            people[j].yspeed = 0

        else:

            people[j].age = 'old'

def peoplepep(people, npeople, npep):

    peplist = random.sample(range(0, (npeople-1)), npep)

    for j in range(npeople):

        people[j].pep = 'nopep'

        

    for i in range(npep):

        j = peplist[i]

        people[j].pep = 'pep'

def peoplesd(people, npeople, npsd):

#    if FREEZE== True:

#        random.seed(30)

#        np.random.seed(30)



    sdlist = random.sample(range(0, (npeople-1)), npsd)

    for j in range(npeople):

        people[j].sd = 'nosd'

        

    for i in range(npsd):

        j = sdlist[i]

        people[j].sd= 'sd'

        
def raninfect(people, npeople, initialinfect):

    inflist = random.sample(range(0, (npeople-1)), initialinfect)

    for i in range(initialinfect):

        j = inflist[i]

        people[j].changecolour("red")
def prob_test(jday,testing, daystarttest, dayendtest, probtest):

    

    if testing == True:

        if jday >= daystarttest and jday <= dayendtest:

            prob = probtest

            pm1 = 1 - prob

        else:

            prob = 0.0

            pm1 = 1 - prob

    else:

        prob = 0.0

        pm1 = 1 - prob

    return prob, pm1

def prob_ppe(jday, daystartppe, dayendppe, probppe):

    

    if jday >= daystartppe and jday <= dayendppe:

        prob = probppe

        pm1 = 1 - prob

    else:

        prob = 0.95

        pm1 = 1 - prob

    return prob, pm1
def prob_sd(jday, daystartsd, dayendsd, probsd):

    

    if jday >= daystartsd and jday <= dayendsd:

        prob = probsd

        pm1 = 1 - prob

    else:

        prob = 0.95

        pm1 = 1 - prob

    return prob, pm1
def prob_ppesd(jday,daystartppe, dayendppe, probppe, daystartsd, dayendsd, probsd):

    

    if jday < daystartppe and jday < daystartsd:

        probppesd = 0.95

        pm1 = 1 - probppesd

        

    elif jday < daystartppe and jday >= daystartsd:

        probppesd = probsd

        pm1 = 1 - probppesd



    elif jday >= daystartppe and jday < daystartsd:

        probppesd = probppe

        pm1 = 1 - probppesd



    elif jday > dayendppe and jday <= dayendsd:

        probppesd = probsd

        pm1 = 1 - probppesd



    elif jday > dayendppe and jday > dayendsd:

        probppesd = 0.95

        pm1 = 1 - probppesd



    elif jday > dayendppe and jday > dayendsd:

        probppesd = probppe

        pm1 = 1 - probppesd

        

    else:

        probppesd = probppe * probsd

        pm1 = 1 - probppesd

    return probppesd, pm1

        
def gridspace(WIDTH,HEIGHT, XBUF,YBUF, ngridx, ngridy):

    xlist = np.linspace(0, WIDTH, ngridx)

    ylist = np.linspace(0, HEIGHT,ngridy)



    xgrid, ygrid = np.meshgrid(xlist, ylist)

    return xgrid, ygrid
def contourplot(xgrid, ygrid, ngridx, ngridy, people, npeople, jday):

    dx = xgrid[0,1] - xgrid[0,0]

    dy = ygrid[1,0] - ygrid[0,0]

    dx2 = dx/2

    dy2 = dy/2

#    ztemp= (300,ngridx,ngridy)

    ztemp= (ngridx,ngridy)



    z = np.zeros((ztemp), dtype = int)



    for i in range(ngridy):

        for j in range(ngridx):

            nred = 0

            xmin = xgrid[i,j]-dx2

            ymin = ygrid[i,j]-dy2

            xmax = xmin + dx

            ymax = ymin + dy

            

            

            for k in range (npeople):

#                print("going thru people", i,j)

                if people[k].colour == "red":

#                    print("found red")

                    if people[k].xcentre >= xmin and people[k].xcentre <= xmax:

#                        print("between xmin and xmax *************************************")

                        if people[k].ycentre >= ymin and people[k].ycentre <= ymax:

#                            print("between ymin and ymax $$$$$$$$$$$$$$$$$")

                            nred += 1

                            print(i,j,nred)



                            z[i,j] = nred



    

    ygridrev = -ygrid              

    us = np.linspace(0, 12, 10, endpoint = True)

#    plt.contourf(xgrid,ygrid,z[jday][:][:], us, cmap ="seismic")

    plt.contourf(xgrid, ygridrev, z, us, cmap ="seismic")



    plt.colorbar()

    plt.xlabel("X - Distance", fontsize = 10)

    plt.ylabel("Y - Distance", fontsize = 10)

    plt.title(" Contour Plot")

    plt.show()

    

    return z

def convergence(jday, y5, people, npeople, nold, ncarehome, ninfected, nqua, nrec, nicu):

    startrun = True

    jdaym14 = jday - 14

    sum1 = 0

    redcount = 0

    nicuyoung = 0

    nrecyoung = 0

    nquayoung = 0

    nquaold = 0

    ngreenyoung = 0

    ngreenold = 0

    ngreencareh= 0

    nicubrown = 0

    ndeath = 0

    

    for i in range(jdaym14, jday-1):

        sum1 = sum1 + y5[i]

    

    if sum1 == 0:

        for i in range(npeople):

            if people[i].colour == "red":

                redcount += 1

            if people[i].colour == "white":

                nicuyoung += 1

            if people[i].colour == "yellow" and people[i].age == "young":

                nrecyoung += 1

            if people[i].colour == "blue" and people[i].age == "young":

                nquayoung += 1

            if people[i].colour == "blue" and people[i].age == "old":

                nquaold += 1

            if people[i].colour == "green" and people[i].age =="young":

                ngreenyoung += 1

            if people[i].colour == "green" and people[i].age =="old":

                ngreenold += 1

            if people[i].colour == "green" and people[i].age =="carehome":

                ngreencareh += 1

            if people[i].colour == "brown":

                nicubrown += 1

            if people[i].colour == "purple":

                ndeath += 1

                



                

        

        if redcount == 0:

            startrun = False

            nicuold = nicu - nicuyoung - nicubrown

            nrecold = nrec - nrecyoung

    

            #print("****CONVERGENCE***")

            #print("jday =", jday, "npeople =", npeople, "nold =", nold, "nyoung =", (npeople-nold))

            #print("ninfected = ", ninfected, "nqua = ", nqua, "nquaold =", nquaold, "nquayoung=", nquayoung)

            #print("nrec = ", nrec, "nrecold =", nrecold, "nrecyoung =", nrecyoung)

            #print("nicu = ", nicu, "nicuold = ", nicuold, "nicucarehome = ", nicubrown, "nicuyoung = ", nicuyoung)     

            #print("ngreenold = ", ngreenold, " ngreenyoung = ", ngreenyoung)

            

            nyoung = npeople - nold

            convergence_results= {"******DAYS TO CONVERGE*******" : jday, 

                    "      ":" ",            

                   "People sample Size" : npeople, 

                   "Number of Elderly" : nold,

                   "Number of young": nyoung,

                   "Number in Elderly in Care Home" : ncarehome,

                   "****":"    ",  

                   "Total Number of Infected" : ninfected,

                   "Total Number of Non-Infected Young People": ngreenyoung,

                   "Total Number of Non-Infected Elderly People": ngreenold,

                   "Total Number of Non-Infected Care Home patients": ngreencareh,

                   "*****":"     ",  

                   "Cummulative Number of people admitted to Qua/Hospital": nqua,

                   "     Number of Elderly in Qua/Hospital on convergence day": nquaold,

                   "     Number of Young in Qua/Hospital on convergence day": nquayoung,

                   "******":"      ",  

                    "Cummulative Number who Recovered": nrec,

                    "    Number of Young who recovered": nrecyoung,

                    "    Number of Elderly who recovered": nrecold,

                    "*******":"       ",  

                    "Cummulative Number in ICU patients on convergence day": nicu,

                    "    Number of Young in ICU": nicuyoung,

                    "    Number of elderly in ICU": nicuold,

                    "    Number of Care Home patient in ICU": nicubrown,  

                    "********":"       ", 

                    "Total Number of Deaths": ndeath}





            print("AT CONVERGENCE:")

            for c in convergence_results:

                format_string = c + ": {" + c + "}" 

                print(format_string.format(**convergence_results))

            

    return startrun  
def Icu_deceased(jday, x1, dailydeath):

    cumdeath = list()

    totaldeath = 0

    for i in range(jday):

        totaldeath = totaldeath + dailydeath[i]

        cumdeath.append(totaldeath)

        

    texttitle = 'Estimated Daily Death  --- Total Death = ' + str (totaldeath)



# Create a figure of size 8x6 inches, 80 dots per inch

    plt.figure(figsize=(8, 6), dpi=80)



# Create a new subplot from a grid of 1x1

#    plt.subplot(1, 1, 1)



    plt.plot(x1, dailydeath, color="yellow", linewidth=2.0, linestyle="-")

    plt.scatter(x1, dailydeath)

    plt.plot(x1, cumdeath, color="red", linewidth=2.0, linestyle="-")

    plt.scatter(x1, cumdeath)

    ax = plt.gca()

    ax.set_facecolor('black')

    plt.grid()

    plt.title(texttitle, fontsize = 20)

    plt.xlabel('Days after first infection', fontsize = 20)

    plt.ylabel('Number of Cases', fontsize = 20)



    plt.show()

    

        

    return totaldeath
def RZeroplot(initialinfect, newinfectday, jday):

    suminfect = initialinfect

    rzero = list()

    for i in range(jday):

        rzero[i] = newinfectday(i)/suminfect

        suminfect = suminfect + newinfectday(i)

    return rzero
def Plot_results(x1,y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11, reference):

    

    fig, axs = plt.subplots(5, 2, figsize=(18,20))

    fig.suptitle(reference, fontsize = 25)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace = None, hspace=0.8)

    axs[0, 0].plot(x1, y1)

    axs[0, 0].scatter(x1,y1)

    axs[0, 0].set_title('Cummulative Infected', fontsize=18)

    axs[0, 0].set_facecolor('black') 

    axs[0, 0].set(xlabel=' ', ylabel='No of cases')



#    plt.grid(True)

#    axs.grid(linestyle='-', linewidth='0.5', color='white')

    axs[1, 0].plot(x1, y2, 'tab:orange')

    axs[1, 0].scatter(x1,y2)

    axs[1, 0].set_title('Cummulative Quarantine Cases', fontsize=18)

    axs[1, 0].set_facecolor('black') 

    axs[1, 0].set(xlabel=' ', ylabel='No of cases')







    axs[3, 0].plot(x1, y3, 'tab:green')

    axs[3, 0].scatter(x1,y3)

    axs[3, 0].set_title('Cummulative ICU Cases', fontsize=18)

    axs[3, 0].set_facecolor('black') 

    axs[3, 0].set(xlabel=' ', ylabel='No of cases')





    axs[2, 0].plot(x1, y4, 'tab:red')

    axs[2, 0].scatter(x1,y4)

    axs[2, 0].set_title('Cummulative Recovered Cases', fontsize=18)

    axs[2, 0].set_facecolor('black') 

    axs[2, 0].set(xlabel=' ', ylabel='No of cases')





    axs[4, 0].plot(x1, y5, 'tab:red')

    axs[4, 0].scatter(x1,y5)

    axs[4, 0].set_title('Daily Infected Cases', fontsize=18)

    axs[4, 0].set_facecolor('yellow') 

    axs[4, 0].set(xlabel=' ', ylabel='No of cases')



    

    axs[0, 1].plot(x1, (y6), 'tab:red')

    axs[0, 1].scatter(x1,y6)

    axs[0, 1].set_title('Daily Effective RZero ', fontsize=18)

    axs[0, 1].set_facecolor('yellow') 

    axs[0, 1].set(xlabel=' ', ylabel='Eff. RZero')





    axs[1, 1].plot(x1, y7, 'tab:red')

    axs[1, 1].scatter(x1,y7)

    axs[1, 1].set_title('Daily Quarantine Admitted Cases', fontsize=18)

    axs[1, 1].set_facecolor('yellow') 

    axs[1, 1].set(xlabel=' ', ylabel='No of cases')





    axs[2, 1].plot(x1, y8, 'tab:red')

    axs[2, 1].scatter(x1,y8)

    axs[2, 1].set_title('Daily ICU admitted Cases ', fontsize=18)

    axs[2, 1].set_facecolor('yellow') 

    axs[2, 1].set(xlabel=' ', ylabel='No of cases')





    axs[3, 1].plot(x1, y9, 'tab:red')

    axs[3, 1].scatter(x1,y9)

    axs[3, 1].set_title('Daily Recovered Cases', fontsize = 18)

    axs[3, 1].set_facecolor('yellow') 

    axs[3, 1].set(xlabel=' ', ylabel='No of cases')



    

#    axs[4, 0].set(xlabel='Days after first Infection', ylabel='No of cases')

    axs[4, 0].set_xlabel("Days after first infection", fontname="Arial", fontsize=18)

    axs[4, 0].set_ylabel("No. of cases", fontname="Arial", fontsize=18)

    

    axs[4, 1].plot(x1, y10, 'tab:red')

    axs[4, 1].scatter(x1,y10)

    axs[4, 1].plot(x1, y11, 'tab:blue')

    axs[4, 1].scatter(x1,y11)



    axs[4, 1].set_title('Daily Quarantine & ICU Occupancy', fontsize=18)

    axs[4, 1].set_facecolor('yellow') 

 

#    axs[4, 1].set(xlabel='Days after first Infection',ylabel='No of cases')

    axs[4, 1].set_xlabel("Days after first infection", fontname="Arial", fontsize=18)

    axs[4, 1].set_ylabel("No. of cases", fontname="Arial", fontsize=18)



    for ax in axs.flat:

#       ax.label_outer()

       ax.grid(linestyle='-', linewidth='1.0', color='green')



    plt.show()

def make_fig():

    

#    markers = ["." , "," , "o" , "v" , "^" , "<", ">"]

    #fig = plt.figure(figsize=(15,10))

    p0 = plt.scatter(x1, y1, marker =",", s=120, zorder = 1, c='r')  # I think you meant this

    p1 = plt.scatter(x2, y2, marker =",",s=120, zorder = 2, c='blue')

    p2 = plt.scatter(x3, y3, marker ="v",s=80, zorder = 3, c='y')

    p3 = plt.scatter(x4, y4, marker ="<",s=80, zorder = 4, c='black')

    p4 = plt.scatter(x5, y5, marker =">",s=80, zorder = 5, c='brown')

    p5 = plt.scatter(x5, y6, marker ="o",s=80, zorder = 6, c='purple')



    plt.legend((p0, p1, p2, p3, p4, p5),

           ('Cum Infected', 'Cum Quarantine', 'Cum Recover', 'Cum ICU', 'Daily Infect', 'RZERO * 10'),

           scatterpoints=1,

           loc='best',

           ncol=3,

           fontsize=20)

    

    plt.grid(linestyle = 'dotted')

    plt.title(' Cummulative Infection & Recovery rate', fontsize = 20)

    plt.xlabel('Days since infection', fontsize = 20)

    plt.ylabel('Cummulative No. of Cases', fontsize = 20)

    #plt.show()
if livedaily == True:

    plt.ion()  # enable interactivity

    %matplotlib auto

    fig = plt.figure(figsize=(15,10))

    #plt.show()

#fig = plt.figure(figsize=(5,5))  # make a figure
x1 = list()

y1 = list()

x2 = list()

y2 = list()

x3 = list()

y3 = list()

x4 = list()

y4 = list()

x5 = list()

y5 = list()

y6 = list()

y7 = list()

y8 = list()

y9 = list()

y10 = list()

y11 = list()

dailydeath = list()

#yrzero[0] = 0



people= []



for i in range(npeople):

    people.append(Person("green", peoplesizex, peoplesizey, peoplespeedx, peoplespeedy))

    

peopleage(people, npeople, nold, ncarehome)    

peoplepep(people, npeople, npep)

peoplesd(people, npeople, npsd)

raninfect(people, npeople, initialinfect)

xgrid, ygrid = gridspace(WIDTH,HEIGHT, XBUF,YBUF, ngridx, ngridy)

   



pinfected = [ninfected]

stepperson =[1]

#############################################################################

testing = covidlogicinput[0]   # To be read from GUI values

if testing == True:

    daystarttest = coviddays[0]

    dayendtest = coviddays[1]

    #probability of Detecting all Infected people through TESTING

    probtest = covidpeopleinput[8]

else:

    daystarttest = 0

    dayendtest = 0

    probtest = 0.0



#The below apply when contact distance is less that size*1.25

ppe = covidlogicinput[1]     # To be read from GUI values

if ppe == True:

    daystartppe = coviddays[2]

    dayendppe = coviddays[3]

    probppe = covidpeopleinput[5]   #Probability of someone wearing PPE (Face Mask) be infected

else:

    daystartppe = 0

    dayendppe = 0

    probppe = 0.95



socdis = covidlogicinput[2]       # To be read from GUI values

if socdis == True:

    daystartsd = coviddays[4]

    dayendsd = coviddays[5]

    #Probability of someone practising Social Distance be infected

    probsd = covidpeopleinput[7]



else:

    daystartsd = 0

    dayendsd = 0

    probsd = 0.95



#Probability of Young to Recover

probyreco = covidpeopleinput[9]

probyreco1 = 1 - probyreco

proboreco = covidpeopleinput[10]

proboreco1 = 1 - proboreco

#Probability of death of ICU patient

probdicu = covidpeopleinput[11]

probdicu1 = 1 - probdicu

    

#Probability of infected carehome resident be sent to ICU when infected

probcarehome = 0.995

probchome1 = 1 - probcarehome



# Probability of getting infected if No PPE and No sd 

nppensd = 0.95

nppensd1 = 1 - nppensd



#############################################################################







jday = 0

nqua= 0

quaday = 0

nicu = 0

icuday = 0

nrec = 0

recday = 0

nquacapa = 0

nicucapa = 0

rzero = 1.0 * 10.0

infectday = initialinfect

startrun = covidlogicinput[5]

reference = covidlogicinput[6]



while startrun:

    x1.append(jday)

    y1.append(ninfected)

    x2.append(jday)

    y2.append(nqua)

    x3.append(jday)

    y3.append(nicu)

    x4.append(jday)

    y4.append(nrec)

    x5.append(jday)

    y5.append(infectday)

    y6.append(rzero)

    

    y7.append(quaday)

    y8.append(icuday)

    y9.append(recday)

    

    infectday = 0

    quaday = 0

    icuday = 0

    recday = 0

    

    

    for j in range(npeople):

        

        people[j].move()

        newpos = people[j].newpos

        if people[j].colour == "red":

            prob, pm1 = prob_test(jday, testing, daystarttest, dayendtest, probtest)

            if np.random.choice(2,1, p=[prob, pm1]) == 0:

                people[j].changecolour("blue")

                people[j].firstquaday = jday

                people[j].movequa(newpos)

            

                nqua += 1

                quaday += 1

#                y2[jday] = nqua

            

        if people[j].colour == "red" and people[j].age == "carehome" and np.random.choice(2,1, p=[probcarehome, probchome1]) == 0:

            people[j].changecolour("brown")

            people[j].moveicu(newpos)

            nicu += 1

            icuday += 1

               

        for i in range(npeople):

   

            d = np.sqrt((people[i].xcentre - people[j].xcentre) **2 + (people[i].ycentre - people[j].ycentre)**2)

            

            if d < contactdist:

                 if people[j].colour == "red" and people[i].colour == "green":

                     if people[i].pep == "pep" and people[i].sd == "sd":

                         prob, pm1 = prob_ppesd(jday,daystartppe, dayendppe, probppe, daystartsd, dayendsd, probsd)

                         if np.random.choice(2,1, p=[prob, pm1]) == 0:

                             people[i].changecolour("red")

                             ninfected += 1

                             infectday += 1

                         

                     if people[i].pep =="pep" and people[i].sd == "nosd":

                         prob, pm1 = prob_ppe(jday, daystartppe, dayendppe, probppe)

                         if np.random.choice(2,1, p=[prob, pm1]) == 0:

                             people[i].changecolour("red")

                             ninfected += 1

                             infectday += 1

                    

                     if people[i].pep == "nopep" and people[i].sd == "sd":

                         prob, pm1 = prob_sd(jday, daystartsd, dayendsd, probsd)

                         if np.random.choice(2,1, p=[prob, pm1]) == 0:

                             people[i].changecolour("red")

                             ninfected += 1

                             infectday += 1

                         

                     if people[i].pep == "nopep" and people[i].sd == "nosd" and np.random.choice(2,1, p=[nppensd, nppensd1]) == 0:

                         people[i].changecolour("red")

                         ninfected += 1

                         infectday += 1





                    

                 if people[j].colour == "green" and people[i].colour == "red":

                     if people[j].pep == "pep" and people[j].sd =="sd":

                         prob, pm1 = prob_ppesd(jday,daystartppe, dayendppe, probppe, daystartsd, dayendsd, probsd)

                         if np.random.choice(2,1, p=[prob, pm1]) == 0:

                             people[j].changecolour("red")

                             ninfected += 1

                             infectday += 1

                         

                     if people[j].pep == "pep" and people[j].sd =="nosd":

                         prob, pm1 = prob_ppe(jday, daystartppe, dayendppe, probppe)

                         if np.random.choice(2,1, p=[prob, pm1]) == 0:

                             people[j].changecolour("red")

                             ninfected += 1

                             infectday += 1

    

                         

                     if people[j].pep == "nopep" and people[j].sd == "sd":

                         prob, pm1 = prob_sd(jday, daystartsd, dayendsd, probsd)

                         if np.random.choice(2,1, p=[prob, pm1]) == 0:

                             people[j].changecolour("red")

                             ninfected += 1

                             infectday += 1

                         

                     if people[j].pep == "nopep" and people[j].sd =="nosd" and np.random.choice(2,1, p=[nppensd, nppensd1]) == 0:

                         people[j].changecolour("red")

                         ninfected += 1

                         infectday += 1





#            y5[jday] = infectday

# Calculation of Effective Rzero

            

            if jday == 0:

                rzero = 1.0 * 10.0

#                rzero =  10.0 * ninfected/y1[jday-1]

            if (y1[jday-1]-y2[jday-1]) == 0:

                rzero = 0.0

            else:   

                rzero = 10.0 *((y1[jday]-y2[jday])/(y1[jday-1]-y2[jday-1]))

         

    tk.update()

    time.sleep(0.1)



    stepperson += [jday]

    pinfected += [ninfected]          

              

    for k in range (npeople):

        if people[k].colour == "blue" and (jday - people[k].firstquaday) >= 14:

            if people[k].age == 'old':

                if np.random.choice(2,1, p=[proboreco1, proboreco]) == 0:

                    people[k].changecolour("black")

                    people[k].moveicu(people[k].newpos)

                    nicu +=1

                    icuday +=1

                else:

                    people[k].changecolour("yellow")

                    people[k].movebac(people[k].newpos)

                    nrec += 1

                    recday +=1

    

            if people[k].age == 'young':

                if np.random.choice(2,1, p=[probyreco1, probyreco]) == 0:

                    people[k].changecolour("white")

                    people[k].moveicu(people[k].newpos)

                    nicu += 1

                    icuday += 1

                else:

                    people[k].changecolour("yellow")

                    people[k].movebac(people[k].newpos)

                    nrec += 1

                    recday += 1

                    

    nquacapa = 0

    nicucapa = 0

    death = 0

    for k in range (npeople):

        if people[k].colour == "blue":

            nquacapa += 1

            

        if people[k].colour == "white" or people[k].colour == "black" or people[k].colour == "brown":

            nicucapa +=1

            prob = np.random.choice(2,1, p=[probdicu, probdicu1])

            if prob == 0:

                death += 1

                people[k].changecolour("purple")

                people[k].xspeed = 0

                people[k].yspeed = 0

                    

    dailydeath.append(death)

    nicucapa = nicucapa - death           

    y10.append(nquacapa)

    y11.append(nicucapa)

            

    if livedaily == True: drawnow(make_fig)

#    drawnow(make_fig2)

    

            

#    z = contourplot(xgrid, ygrid, ngridx, ngridy, people, npeople, jday)

#    print(z)

    daylabel(jday)

    jday +=1

    if jday > 15:

        startrun = convergence(jday, y5, people, npeople, nold, ncarehome, ninfected, nqua, nrec, nicu)



for i in range(jday):

    y6[i] = y6[i]/10

    

   

#%matplotlib inline

Plot_results(x1,y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,reference)

#%matplotlib inline

totaldeath = Icu_deceased(jday, x1, dailydeath)

tk.mainloop()    