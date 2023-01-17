import numpy as np

import sqlite3 as sq

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from PIL import Image

import os

import datetime

from tkinter import *

from tkinter.ttk import *

from tkinter import messagebox

import tkinter.font as font

from tkinter import messagebox

from tkinter import ttk

from win32api import GetSystemMetrics

import socket  

import ipaddress

import webbrowser

import datetime

from datetime import date, timedelta

from urllib.request import urlopen

from bs4 import BeautifulSoup

from PIL import Image, ImageTk



t1=datetime.datetime.now()

fcol="SeaGreen1"

w1="blue violet"

w2="orange red"

w3="gold"

w4="lime green"

w5='orchid1'

w6='salmon'

w7='purple4'

w8='yellow2'

w9='brown1'

w10='blue'

w11='dodger blue'

font1="Bookman Old Style Bold"

font2="Bookman Old Style bold Italic"

font3="Bookman Old Style bold"



conn = sq.connect('velvet.db')

print("Opened database successfully")

try:

    conn.execute('''CREATE TABLE USER

         (ID INTEGER PRIMARY KEY AUTOINCREMENT,

         NAME           TEXT    NOT NULL,

         EMAIL          TEXT     NOT NULL,

         PASS        CHAR(50),

         ADDRESS     CHAR(100),

         AGE         CHAR(50));''')

    print("USER Table created successfully")

    conn.execute('''CREATE TABLE USERACCNT

         (ID INTEGER PRIMARY KEY AUTOINCREMENT,

         NAME           TEXT    NOT NULL,

         EMAIL          TEXT     NOT NULL,

         PASS        CHAR(50),

         CREDCARD    INT   NOT NULL,

         BALANCE     FLOAT   NOT NULL);''')

    print("USER Table created successfully")

    conn.execute('''CREATE TABLE PRODUCT

         (ID INTEGER PRIMARY KEY AUTOINCREMENT,

         NAME           TEXT    NOT NULL,

         CAKETYPE           TEXT     NOT NULL,

         PRICE        FLOAT,

         WEIGHT       FLOAT,

         FLAVOUR      CHAR(50),

         CAKEID       INT,

         CSHAPE        CHAR(20));''')

    print("PRODUCT Table created successfully")

    conn.execute('''CREATE TABLE USERORDER

         (ID INTEGER PRIMARY KEY AUTOINCREMENT,

         NAME           TEXT    NOT NULL,

         TYPE           TEXT     NOT NULL,

         PRICE        FLOAT,

         DATE      CHAR(12),

         DELIVDATE    CHAR(12),

         ADDRESS    CHAR(100),

         ITEM     CHAR(100));''')

    print("PRODUCT Table created successfully")

except:

    pass
def adminpanel(name):

    admn=Tk()

    admn.title("Admin")

    admn.configure(bg='tomato')

    style = Style() 

    style.configure('W.TButton', font = 

               (font1, 20, 'bold', 'italic'), 

                foreground = w2,borderwidth = '4') 

    admn.geometry(str(GetSystemMetrics(0))+'x'+str(GetSystemMetrics(1)))

    lbl1=Label(admn, text="Velvet Administrator Panel",font=(font1,40))

    lbl1.place(x=390,y=10)

    lbl1.configure(foreground='white',background='tomato')

    

    additem=Label(admn, text="Insert Item",font=(font2,25))

    additem.place(x=520,y=190)

    additem.configure(foreground='white',background='tomato')

    

    def additems():

        addit=Tk()

        addit.title("Add Items")

        addit.configure(bg='tomato')

        style = Style() 

        style.configure('W.TButton', font = 

                   ('Times New Roman', 20, 'bold', 'italic'), 

                    foreground = 'red',borderwidth = '4') 

        addit.geometry('600x600')

        lbl1=Label(addit, text="Add Items",font=(font1,40))

        lbl1.place(x=200,y=10)

        lbl1.configure(foreground='white',background='tomato')

        

        additem=Label(addit, text="Cake Name",font=(font2,20))

        additem.place(x=50,y=100)

        additem.configure(foreground='white',background='tomato')

        additeme=Entry(addit,width=25)

        additeme.place(x=260,y=110)

        

        ittyp=Label(addit, text="Cake Type",font=(font2,20))

        ittyp.place(x=50,y=160)

        ittyp.configure(foreground='white',background='tomato')

        ittypc = Combobox(addit,width=22)

        ittypc['values']= ["Cream Cake","Dry Cake","Jar Cake"]

        ittypc.current(0)

        ittypc.place(x=260,y=170)

        

        prc=Label(addit, text="Cake Price",font=(font2,20))

        prc.place(x=50,y=220)

        prc.configure(foreground='white',background='tomato')

        prce=Entry(addit,width=25)

        prce.place(x=260,y=230)

        

        wgt=Label(addit, text="Cake Weight",font=(font2,20))

        wgt.place(x=50,y=280)

        wgt.configure(foreground='white',background='tomato')

        wgte=Entry(addit,width=25)

        wgte.place(x=260,y=290)

        

        flv=Label(addit, text="Cake Flavour",font=(font2,20))

        flv.place(x=50,y=340)

        flv.configure(foreground='white',background='tomato')

        flvc = Combobox(addit,width=22)

        flvc['values']= ["Chocolate"]

        flvc.current(0)

        flvc.place(x=260,y=350)

        

        ckid=Label(addit, text="Cake ID",font=(font2,20))

        ckid.place(x=50,y=400)

        ckid.configure(foreground='white',background='tomato')

        ckide=Entry(addit,width=25)

        ckide.place(x=260,y=410)

        

        cshp=Label(addit, text="Cake Shape",font=(font2,20))

        cshp.place(x=50,y=460)

        cshp.configure(foreground='white',background='tomato')

        cshpc = Combobox(addit,width=22)

        cshpc['values']= ["Circle","Square","Oval"]

        cshpc.current(0)

        cshpc.place(x=260,y=470)

        

        def insitem():

            val=(additeme.get(),ittypc.get(),prce.get(),wgte.get(),flvc.get(),ckide.get(),cshpc.get())

            sql="INSERT INTO PRODUCT (NAME,CAKETYPE,PRICE,WEIGHT,FLAVOUR,CAKEID,CSHAPE) VALUES (?,?,?,?,?,?,?)"

            conn.execute(sql,val);

            conn.commit()

            messagebox.showinfo('Add Item Info', 'Cake ID: '+ckide.get()+" Added Successfully")

        insitembtn = Button(addit, text="Add", command=insitem, style = 'W.TButton')

        insitembtn.place(x=170,y=510)

        

        def refr():

            addit.destroy()

            additems()

        clearbtn = Button(addit, text="Reload Page", command=refr, style = 'W.TButton')

        clearbtn.place(x=250,y=510)

        addit.mainloop()

    

    additembtn = Button(admn, text="Add Item", command=additems, style = 'W.TButton')

    additembtn.place(x=840,y=190)

    

    chkcustrec=Label(admn, text="View Records",font=(font2,25))

    chkcustrec.place(x=520,y=260)

    chkcustrec.configure(foreground='white',background='tomato')

    

    def checkcustrec():

        chkit=Tk()

        chkit.title("Customer Record")

        chkit.geometry('900x500')

        lbl1=Label(chkit, text="Customer Record",font=(font1,40))

        lbl1.place(x=250,y=10)

        lbl1.configure(foreground=w2)

        cols = ('Name', 'Email','Address','Age')

        listBox = ttk.Treeview(chkit, columns=cols, show='headings')

        listBox.place(x=600,y=120)

        for col in cols:

            listBox.heading(col, text=col)    

            listBox.place(x=20,y=100)

        cur6=conn.execute("SELECT NAME,EMAIL,ADDRESS,AGE FROM USER")

        curdetails=[]

        for row in cur6:

            listBox.insert("", "end", values=row)

        

        def checkcustrecclose():

            chkit.destroy()

            

        close = Button(chkit, text="Back", command=checkcustrecclose, style = 'W.TButton',width=18)

        close.place(x=350,y=330)



        chkit.mainloop()

    

    chkcustrecbtn = Button(admn, text="Details", command=checkcustrec, style = 'W.TButton')

    chkcustrecbtn.place(x=840,y=260)

    

    purrec=Label(admn, text="View Purchase",font=(font2,25))

    purrec.place(x=520,y=330)

    purrec.configure(foreground='white',background='tomato')

    

    def purhist():

        purhit=Tk()

        purhit.geometry(str(GetSystemMetrics(0))+'x'+str(GetSystemMetrics(1)))

        lbl1=Label(purhit, text="Purchase History",font=(font1,40))

        lbl1.place(x=550,y=10)

        lbl1.configure(foreground=w2)

        cols = ('Name', 'Type','Price','Order Date','Delivery Date','Address','Cake ID')

        listBox = ttk.Treeview(purhit, columns=cols, show='headings')

        listBox.place(x=600,y=120)

        for col in cols:

            listBox.heading(col, text=col)    

            listBox.place(x=20,y=100)

        cur4=conn.execute("SELECT * FROM USERORDER")

        curdetails=[]

        for row in cur4:

            listBox.insert("", "end", values=row)



        def trackclose():

            purhit.destroy()

            

        close = Button(purhit, text="Back", command=trackclose, style = 'W.TButton',width=18)

        close.place(x=620,y=400)

        

        def purchstat():

            cur5=conn.execute("SELECT * FROM USERORDER")

            typ=[]

            for row in cur5:

                typ.append(row[2])

            

            try:

                df=pd.DataFrame({"Type":typ})

                plt.figure(figsize=(10,5))

                plt.title("Statistics by Cake Type",fontsize=18,color="b")

                sns.countplot(df['Type'],color='m')

                plt.savefig("Stat_Type.png")

                plt.figure(figsize=(10,5))

                image0=Image.open("Stat_Type.png")

                image0.show(image0) 

            except:

                pass

        

        purstat = Button(purhit, text="Details", command=purchstat, style = 'W.TButton',width=18)

        purstat.place(x=330,y=400)

        

        purhit.mainloop()

    

    purrecbtn = Button(admn, text="Details", command=purhist, style = 'W.TButton')

    purrecbtn.place(x=840,y=330)

    

    earn=Label(admn, text="Check Revenue",font=(font2,25))

    earn.place(x=520,y=400)

    earn.configure(foreground='white',background='tomato')

    

    def earnstat():

        #estat=Tk()

        cur7=conn.execute("SELECT * FROM USERORDER")

        pric=[]

        for row in cur7:

                pric.append(row[2])

                

        df=pd.DataFrame({"Price":pric})

        plt.figure(figsize=(10,5))

        plt.title("Revenue Statistics",fontsize=18,color="b")

        plt.plot(pric)

        plt.plot(pric,"h")

        plt.savefig("Stat_Price.png")

        image1 = Image.open("Stat_Price.png")

        image1.show(image1)     

        

    

    purrecbtn = Button(admn, text="Details", command=earnstat, style = 'W.TButton')

    purrecbtn.place(x=840,y=400)

    

    admn.mainloop()

#adminpanel('admin@gmail.com')
def track_order(usname):

    tr=Tk()

    tr.title("Track Orders")

    tr.configure(bg='white')

    tr.geometry(str(GetSystemMetrics(0))+'x'+str(GetSystemMetrics(1)))

    lbl1=Label(tr, text="Order History",font=(font1,40))

    lbl1.place(x=550,y=10)

    lbl1.configure(foreground=w1,background='white')

    

    cols = ('Name', 'Type','Price','Order Date','Delivery Date','Address','Cake ID')

    listBox = ttk.Treeview(tr, columns=cols, show='headings')

    listBox.place(x=600,y=120)

    for col in cols:

        listBox.heading(col, text=col)    

        listBox.place(x=20,y=100)

    

    cur3=conn.execute("SELECT * FROM USERORDER where NAME='"+usname+"'")

    curdetails=[]

    for row in cur3:

        listBox.insert("", "end", values=row[1:])

    

    def trackclose():

        tr.destroy()

    close = Button(tr, text="Back", command=trackclose, style = 'W.TButton',width=18)

    close.place(x=700,y=400)

    

    tr.mainloop()

#track_order('Subha')
def purchase(curuser,curmail):

    logged=curuser

    logmail=curmail

    pur=Tk()

    pur.title("Velvet Chocolate")

    pur.configure(bg='misty rose')

    '''image1 = Image.open("C04.jpg")

    tkpi = ImageTk.PhotoImage(image1)

    label_image = Label(pur, image=tkpi)

    label_image.place(x=0,y=0)'''

    style = Style() 

    style.configure('W.TButton', font = 

               ('Times New Roman', 14, 'bold', 'italic'), 

                foreground = 'deep pink',borderwidth = '4')

    

    pur.geometry(str(GetSystemMetrics(0))+'x'+str(GetSystemMetrics(1)))

    lbl1=Label(pur, text="Order Item",font=(font1,40))

    lbl1.place(x=550,y=10)

    lbl1.configure(foreground='deep pink',background='misty rose')

    

    bid=Label(pur, text="User: "+logged[0],font=(font2,10))

    bid.place(x=1200,y=10)

    bid.configure(foreground=w2,background='misty rose')

    bid=Label(pur, text="Logged in as: "+logmail[0],font=(font2,10))

    bid.place(x=1200,y=10)

    bid.configure(foreground=w9,background='misty rose')

    

    nm=[]

    ty=[]

    fl=[]

    sh=[]

    cur=conn.execute("SELECT * FROM PRODUCT")

    for row in cur:

            nm.append(row[1])

            ty.append(row[2])

            fl.append(row[5])

            sh.append(row[7])

    name=Label(pur, text="Cake Name",font=(font2,20))

    name.place(x=10,y=120)

    name.configure(foreground='deep pink',background='misty rose')

    namec = Combobox(pur,width=20)

    namec['values']=tuple(np.unique(np.array(nm)))

    namec.current(0)

    namec.place(x=190,y=130)

    

    typ=Label(pur, text="Cake Type",font=(font2,20))

    typ.place(x=340,y=120)

    typ.configure(foreground='deep pink',background='misty rose')

    typc = Combobox(pur,width=20)

    typc['values']= tuple(np.unique(np.array(ty)))

    typc.current(1)

    typc.place(x=510,y=130)

    

    flv=Label(pur, text="Cake Taste",font=(font2,20))

    flv.place(x=10,y=190)

    flv.configure(foreground='deep pink',background='misty rose')

    flvc = Combobox(pur,width=20)

    flvc['values']= tuple(np.unique(np.array(fl)))

    flvc.current(0)

    flvc.place(x=190,y=200)

    

    shp=Label(pur, text="Cake Size",font=(font2,20))

    shp.place(x=340,y=190)

    shp.configure(foreground='deep pink',background='misty rose')

    shpc = Combobox(pur,width=20)

    shpc['values']= tuple(np.unique(np.array(sh)))

    shpc.current(0)

    shpc.place(x=510,y=200)

    

        

    def search():

        cols = ('Name', 'Type','Price','Weight','Flavour','Cake ID','Shape')

        listBox = ttk.Treeview(pur, columns=cols, show='headings')

        listBox.place(x=600,y=120)

        for col in cols:

            listBox.heading(col, text=col)    

            listBox.place(x=20,y=450)

        cursor = conn.execute("SELECT * from PRODUCT where CAKETYPE='"+typc.get()+"'")

        details=[]

        for row in cursor:

            listBox.insert("", "end", values=row[1:])



    srch = Button(pur, text="Search", command=search,style = 'W.TButton',width=9)

    srch.place(x=290,y=250)

    

    bid=Label(pur, text="Order your Cake",font=(font2,20))

    bid.place(x=750,y=140)

    bid.configure(foreground='deep pink',background='misty rose')

    bid=Label(pur, text="Cake ID",font=(font2,15))

    bid.place(x=700,y=190)

    bid.configure(foreground='deep pink',background='misty rose')

    bent=Entry(pur,width=12)

    bent.place(x=810,y=196)

    ship = Combobox(pur,width=13)

    ship['values']= ["Sky Supplier","Speed Suppliers","ElasticSupply"]

    ship.current(1)

    ship.place(x=900,y=196)

         

    

    def bookpurnorm():

        if ship.get()=="Sky Supplier":

            cur1=conn.execute("SELECT NAME,ADDRESS,EMAIL FROM USER where EMAIL='"+logmail[0]+"'")

            curdetails=[]

            for row in cur1:

                curdetails.append(list(row))

            print(curdetails)



            cakedetails=[]

            cur2=conn.execute("SELECT CAKETYPE,PRICE FROM PRODUCT where CAKEID='"+bent.get()+"'")

            for row in cur2:

                cakedetails.append(list(row))

            print(cakedetails)



            val=(curdetails[0][0],cakedetails[0][0],str(int(cakedetails[0][1])+30),date.today(),date.today() + timedelta(4),curdetails[0][1],bent.get())

            sql="INSERT INTO USERORDER (NAME,TYPE,PRICE,DATE,DELIVDATE,ADDRESS,ITEM) VALUES (?,?,?,?,?,?,?)"

            conn.execute(sql,val);

            conn.commit()         

            

            cur9=conn.execute("SELECT BALANCE FROM USERACCNT WHERE EMAIL='"+logmail[0]+"'")

            b=[]

            for row in cur9:

                b.append(row)

            lastbl=b[0][0]

            updbl=lastbl-int(cakedetails[0][1])+30

            print("Updated Balance: ",updbl)

            conn.execute("UPDATE USERACCNT SET BALANCE='"+str(updbl)+"' WHERE EMAIL='"+logmail[0]+"'")

            conn.commit()

            

            messagebox.showinfo('Order Info', 'Placed order with ID'+bent.get())

        if ship.get()=="Speed Suppliers":

            cur1=conn.execute("SELECT NAME,ADDRESS FROM USER where EMAIL='"+logmail[0]+"'")

            curdetails=[]

            for row in cur1:

                curdetails.append(list(row))



            cakedetails=[]

            cur2=conn.execute("SELECT CAKETYPE,PRICE FROM PRODUCT where CAKEID='"+bent.get()+"'")

            for row in cur2:

                cakedetails.append(list(row))



            val=(curdetails[0][0],cakedetails[0][0],str(int(cakedetails[0][1])+40),date.today(),date.today() + timedelta(4),curdetails[0][1],bent.get())

            sql="INSERT INTO USERORDER (NAME,TYPE,PRICE,DATE,DELIVDATE,ADDRESS,ITEM) VALUES (?,?,?,?,?,?,?)"

            conn.execute(sql,val);

            conn.commit()

            messagebox.showinfo('Order Info', 'Placed order with ID'+bent.get())

        if ship.get()=="ElasticSupply":

            cur1=conn.execute("SELECT NAME,ADDRESS FROM USER where EMAIL='"+logmail[0]+"'")

            curdetails=[]

            for row in cur1:

                curdetails.append(list(row))



            cakedetails=[]

            cur2=conn.execute("SELECT CAKETYPE,PRICE FROM PRODUCT where CAKEID='"+bent.get()+"'")

            for row in cur2:

                cakedetails.append(list(row))



            val=(curdetails[0][0],cakedetails[0][0],str(int(cakedetails[0][1])+50),date.today(),date.today() + timedelta(4),curdetails[0][1],bent.get())

            sql="INSERT INTO USERORDER (NAME,TYPE,PRICE,DATE,DELIVDATE,ADDRESS,ITEM) VALUES (?,?,?,?,?,?,?)"

            conn.execute(sql,val);

            conn.commit()

            messagebox.showinfo('Order Info', 'Placed order with ID'+bent.get())

    

    bookn = Button(pur, text="Normal Delivery", command=bookpurnorm, style = 'W.TButton',width=14)

    bookn.place(x=700,y=240)

    

    def bookpururg():

        cur1=conn.execute("SELECT NAME,ADDRESS FROM USER where EMAIL='"+logmail[0]+"'")

        curdetails=[]

        for row in cur1:

            curdetails.append(list(row))

        

        cakedetails=[]

        cur2=conn.execute("SELECT CAKETYPE,PRICE FROM PRODUCT where CAKEID='"+bent.get()+"'")

        for row in cur2:

            cakedetails.append(list(row))

        

        val=(curdetails[0][0],cakedetails[0][0],str(int(cakedetails[0][1])+120),date.today(),date.today() + timedelta(2),curdetails[0][1],bent.get())

        sql="INSERT INTO USERORDER (NAME,TYPE,PRICE,DATE,DELIVDATE,ADDRESS,ITEM) VALUES (?,?,?,?,?,?,?)"

        conn.execute(sql,val);

        conn.commit()

        messagebox.showinfo('Order Info', 'Placed order with ID'+bent.get())

    

    bookur = Button(pur, text="Urgent Delivery", command=bookpururg, style = 'W.TButton',width=13)

    bookur.place(x=870,y=240)

    

    trk=Label(pur, text="Check Status",font=(font2,20))

    trk.place(x=1150,y=170)

    trk.configure(foreground='deep pink',background='misty rose')

    

    def trcakord():

        track_order(logged[0])

        

    trackorder = Button(pur, text="Track Item", command=trcakord, style = 'W.TButton',width=13)

    trackorder.place(x=1100,y=240)

    

    '''enq=Label(pur, text="Enquire Order",font=(font2,20))

    enq.place(x=1200,y=240)

    enq.configure(foreground=w2,background='white')'''

    

    def enquire():

        track_order(logged[0])

        

    enqorder = Button(pur, text="Enquire Order", command=enquire, style = 'W.TButton',width=13)

    enqorder.place(x=1300,y=240)

    

    pur.mainloop()

#purchase("Stephen George","step@gmail.com")
def aboutus():

    about=Tk()

    about.configure(bg='floral white')

    about.title("Velvet Chocolate")

    about.geometry('1200x200')

    lbl1=Label(about, text="About Us",font=(font1,25))

    lbl1.place(x=500,y=10)

    lbl1.configure(foreground=w1,background='floral white')

    

    t1='''DELIVER CHOCOLATES, CAKES & COOKIES TO YOUR LOVED ONES ANYWHERE ACROSS INDIA WITH VELVET FINE CHOCOLATES!'''

    t2='''Chocolates, cookies and cakes have long been accustomed to creating vibrant environments whether its home or offices.'''

    t3=''' We bring you some of the finest collection of chocolate gifts that'''

    t4='''are available in various varieties. Apart from that, we also deliver custom chocolate gifts online'''

    t5='''to any location across India. We make your occasions fulfilling whether you are geographically '''

    t6='''close to your loved ones or not. One can order chocolate gift boxes online from miles apart and still feel'''

    t7='''the connection. We ensure that every order placed is well refined with the smallest '''

    t8='''details and delivered across any locations.'''

    

    lbl2=Label(about, text="                             ",font=(font1,12))

    lbl2.place(x=50,y=70)

    lbl2.configure(text=t1,foreground=w2,background='floral white')

    

    lbl3=Label(about, text="                            ",font=(font1,10))

    lbl3.place(x=50,y=100)

    lbl3.configure(text=t2+t3,background='floral white')

    

    lbl4=Label(about, text="                            ",font=(font1,10))

    lbl4.place(x=50,y=120)

    lbl4.configure(text=t4+t5,background='floral white')

    

    lbl5=Label(about, text="                            ",font=(font1,10))

    lbl5.place(x=50,y=140)

    lbl5.configure(text=t6+t7,background='floral white')

    

    lbl6=Label(about, text="                            ",font=(font1,10))

    lbl6.place(x=50,y=160)

    lbl6.configure(text=t8,background='floral white')

    about.mainloop()



def contactus():

    cont=Tk()

    cont.configure(bg='white')

    cont.title("Velvet Chocolate")

    cont.geometry('690x400')

    l1=Label(cont, text="Contact Information",font=(font1,25))

    l1.place(x=200,y=10)

    l1.configure(foreground=w7,background='white')

    

    tl1='''Address: '''

    t10='''Unit No 22, Shah Industrial Estate, Deonar Village Road, Deonar, Govandi East. Mumbai:400088'''

    tl2='''Mobile: '''

    t11='''9819800995'''

    tl3='''Email ID: '''

    t12='''velvetfinechocolates@gmail.com'''

    tl4='''Alternate Email ID: '''

    t13='''velvetfinechocolates1@gmail.com'''

    

    l2=Label(cont, text="                             ",font=(font1,10))

    l2.place(x=50,y=70)

    l2.configure(text=tl1,foreground=w7,background='white')

    

    l3=Label(cont, text="                            ",font=(font1,10))

    l3.place(x=50,y=100)

    l3.configure(text=t10,foreground=w2,background='white')

    

    l2=Label(cont, text="                             ",font=(font1,10))

    l2.place(x=50,y=130)

    l2.configure(text=tl2,foreground=w7,background='white')

    

    l3=Label(cont, text="                            ",font=(font1,10))

    l3.place(x=50,y=160)

    l3.configure(text=t11,foreground=w2,background='white')

    

    l2=Label(cont, text="                             ",font=(font1,10))

    l2.place(x=50,y=190)

    l2.configure(text=tl3,foreground=w7,background='white')

    

    l3=Label(cont, text="                            ",font=(font1,10))

    l3.place(x=50,y=220)

    l3.configure(text=t12,foreground=w2,background='white')

    

    l2=Label(cont, text="                             ",font=(font1,10))

    l2.place(x=50,y=250)

    l2.configure(text=tl4,foreground=w7,background='white')

    

    l3=Label(cont, text="                            ",font=(font1,10))

    l3.place(x=50,y=280)

    l3.configure(text=t13,foreground=w2,background='white')

        

    cont.mainloop()



def register():

    regs=Tk()

    regs.configure(bg='SeaGreen1')

    regs.title("New Registration")

    regs.geometry('440x340')

    lb1=Label(regs, text="New Registration",font=(font1,20))

    lb1.place(x=80,y=5)

    lb1.configure(foreground=w9,background='SeaGreen1')

    

    uname=Label(regs, text="Name",font=(font1,10))   

    uname.place(x=30,y=60)

    uname.configure(foreground=w9,background='SeaGreen1')

    unamee=Entry(regs,width=20)

    unamee.place(x=160,y=60)

    

    umail=Label(regs, text="E-Mail",font=(font1,10))  

    umail.place(x=30,y=90)

    umail.configure(foreground=w9,background='SeaGreen1')

    umaile=Entry(regs,width=20)

    umaile.place(x=160,y=90)

    

    upass=Label(regs, text="Password",font=(font1,10))

    upass.place(x=30,y=120)

    upass.configure(foreground=w9,background='SeaGreen1')

    upasse=Entry(regs,width=20)

    upasse.place(x=160,y=120)

    

    uaddr=Label(regs, text="Address",font=(font1,10))   

    uaddr.place(x=30,y=150)

    uaddr.configure(foreground=w9,background='SeaGreen1')

    uaddre=Entry(regs,width=20)

    uaddre.place(x=160,y=150)



    uage=Label(regs, text="Age",font=(font1,10)) 

    uage.place(x=30,y=180)

    uage.configure(foreground=w9,background='SeaGreen1')

    uagee=Entry(regs,width=20)

    uagee.place(x=160,y=180)

    

    visa=Label(regs, text="Credit Card",font=(font1,10))   

    visa.place(x=30,y=210)

    visa.configure(foreground=w9,background='SeaGreen1')

    visae=Entry(regs,width=20)

    visae.place(x=160,y=210)

    

    amnt=Label(regs, text="Credit Amount",font=(font1,10))  

    amnt.place(x=30,y=240)

    amnt.configure(foreground=w9,background='SeaGreen1')

    amnte=Entry(regs,width=20)

    amnte.place(x=160,y=240)

    

    def auth():

        cursor = conn.execute("SELECT * from USER")

        email=[]

        for row in cursor:

            email.append(row[2])

        print(email)

        

        if unamee.get()!="" or umaile.get()!="" or upasse.get()!="":



            if umaile.get() in email:

                messagebox.showerror('Info', 'Email ID Exists')

                regs.destroy()

                register()



            if umaile.get() not in email:

                val1=(unamee.get(),umaile.get(),upasse.get(),uaddre.get(),uagee.get())

                sql1="INSERT INTO USER (NAME,EMAIL,PASS,ADDRESS,AGE) VALUES (?,?,?,?,?)"

                conn.execute(sql1,val1);

                conn.commit()

                

                val2=(unamee.get(),umaile.get(),upasse.get(),visae.get(),amnte.get())

                sql2="INSERT INTO USERACCNT (NAME,EMAIL,PASS,CREDCARD,BALANCE) VALUES (?,?,?,?,?)"

                conn.execute(sql2,val2);

                conn.commit()

                

                messagebox.showinfo('VisaCheck', 'Visa Chcek Successful for '+visae.get()+'\n$'+amnte.get()+' Added Successfully')

                messagebox.showinfo('Info', 'Registration Successful')



        else:

            messagebox.showerror('Info', 'Registration Failed')

            regs.destroy()

            register()



        

    user = Button(regs, text="Register", command=auth,style = 'W.TButton',width=9)

    user.place(x=165,y=280)

    

    def refrsh():

        regs.destroy()

        register()

    refrs = Button(regs, text="Reload", command=refrsh,style = 'W.TButton',width=9)

    refrs.place(x=340,y=120)

    

    

    def clse():

        regs.destroy()

    

    clos = Button(regs, text="Cancel", command=clse,style = 'W.TButton',width=9)

    clos.place(x=340,y=160)

    

    regs.mainloop()



def start_page():

    windows = Tk()

    #windows.configure(bg='white')

    image1 = Image.open("C04.jpg")

    tkpi = ImageTk.PhotoImage(image1)

    label_image = Label(windows, image=tkpi)

    label_image.place(x=0,y=0)

    style = Style() 

    style.configure('W.TButton', font = 

               ('Times New Roman', 20, 'bold', 'italic'), 

                foreground = w9,borderwidth = '4') 

    wid=GetSystemMetrics(0)

    hgt=GetSystemMetrics(1)

    windows.title("Velvet")

    windows.geometry(str(wid)+'x'+str(hgt))

    

       

    show=Label(windows, text="Velvet Fine Chocolate",font=(font1,45))

    show.place(x=10,y=10)

    show.configure(background='white',foreground=w9)

    

    def checknet():

        webbrowser.open_new_tab('https://www.velvetfinechocolates.com/')

    btn1 = Button(windows, text="Website", command=checknet,style = 'W.TButton',width=8)

    btn1.place(x=1200,y=50)

    

    def velvet():

        vel=Tk()

        vel.configure(bg='SeaGreen1')

        vel.title("Login Portal")

        vel.geometry('650x400')

        l1=Label(vel, text="Login Portal",font=(font1,20))

        l1.place(x=250,y=5)

        l1.configure(foreground=w9,background='SeaGreen1')

        

        lab1=Label(vel, text="User",font=(font1,14))

        lab1.place(x=120,y=70)

        lab1.configure(foreground=w9,background='SeaGreen1')

        uidl=Label(vel, text="EMAIL ID",font=(font1,10))

        uidl.place(x=30,y=120)

        uidl.configure(foreground=w9,background='SeaGreen1')

        uid=Entry(vel,width=20)

        uid.place(x=130,y=120)

        upassl=Label(vel, text="PASSWORD",font=(font1,10))

        upassl.place(x=30,y=150)

        upassl.configure(foreground=w9,background='SeaGreen1')

        upass=Entry(vel,width=20,show='*')

        upass.place(x=130,y=150)

        

        lab2=Label(vel, text="Admin",font=(font1,14))

        lab2.place(x=450,y=70)

        lab2.configure(foreground=w9,background='SeaGreen1')

        aidl=Label(vel, text="EMAIL ID",font=(font1,10))

        aidl.place(x=400,y=120)

        aidl.configure(foreground=w9,background='SeaGreen1')

        aid=Entry(vel,width=20)

        aid.place(x=500,y=120)

        apassl=Label(vel, text="PASSWORD",font=(font1,10))

        apassl.place(x=400,y=150)

        apassl.configure(foreground=w9,background='SeaGreen1')

        apass=Entry(vel,width=20,show='*')

        apass.place(x=500,y=150)

        

        def ulogin():

            cursor = conn.execute("SELECT * from USER")

            email=[]

            passw=[]

            nameu=[]

            for row in cursor:

                nameu.append(row[1])

                email.append(row[2])

                passw.append(row[3])

            

            if uid.get() in email:

                ind=email.index(uid.get())

                pss=passw[ind]

                if pss==upass.get():

                    curuser=[nameu[ind]]

                    curmail=[email[ind]]

                    print(curuser)

                    messagebox.showinfo('Info', 'Login Successful')

                    vel.destroy()

                    purchase(curuser,curmail)

                else:

                    messagebox.showerror('Info', 'Login Failed')

            

        

        user = Button(vel, text="Login", command=ulogin,style = 'W.TButton',width=9)

        user.place(x=130,y=180)

        

        def adlogin():

            if aid.get()=="admin@gmail.com" and apass.get()=="12345":

                aid.configure(text="")

                apass.configure(text="")

                messagebox.showinfo('Info', 'Login Successful')

                adminpanel("admin@gmail.com")

            else:

                messagebox.showerror('Info', 'Login Error')

                

        

        adm = Button(vel, text="Login", command=adlogin,style = 'W.TButton',width=9)

        adm.place(x=500,y=180)

        

        show=Label(vel, text="Click to Register",font=(font2,18))

        show.place(x=130,y=250)

        show.configure(foreground='violet red',background='SeaGreen1')

        

        def reg():

            vel.destroy()

            register()

        

        btn1 = Button(vel, text="Register", command=reg,style = 'W.TButton',width=15)

        btn1.place(x=370,y=254)

        

        vel.mainloop()

        

    

    btn1 = Button(windows, text="Velvet", command=velvet,style = 'W.TButton',width=7)

    btn1.place(x=1376,y=50)



    

    windows.mainloop()

start_page()