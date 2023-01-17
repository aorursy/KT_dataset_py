import sqlite3 as sq

sq.sqlite_version
import matplotlib.pyplot as plt
conn=sq.connect('../input/database.sqlite') #importing sql database

cur=conn.cursor() #defining the python cursor

#database contains column names of id, name, year, gender, count
cur.execute("SELECT year, count FROM NationalNames WHERE name='Lillian' AND gender='F';") #executes the SQL command to return the year and count of females named Lillian.

data=cur.fetchall() #stores the return 

x_val = [x[0] for x in data] #makes a list of the year values as x

y_val = [x[1] for x in data] #makes of list of the count values as y

fig=plt.figure(figsize=(5,5))

plt.plot(x_val,y_val, label="Name=Lillian, Gender=F") #plots x and y

plt.legend(fontsize=12)

ax=fig.add_subplot(111)

ax.set_xlabel('Year', fontsize=12)

ax.set_ylabel('Number of people', fontsize=12)

plt.show()
cur.execute("SELECT name FROM NationalNames WHERE year='2014' AND count<=10 LIMIT 20;")

cur.fetchall()
cur.execute("SELECT DISTINCT name FROM NationalNames WHERE name LIKE 'L%' LIMIT 20;")

cur.fetchall()
cur.execute("SELECT year, MAX(LENGTH(name)) FROM NationalNames WHERE gender='F' GROUP BY year;") #max length of female name by year

longF=cur.fetchall()

cur.execute("SELECT year, MAX(LENGTH(name)) FROM NationalNames WHERE gender='M' GROUP BY year;") #max length of male name by year

longM=cur.fetchall()

xF = [x[0] for x in longF]

yF = [x[1] for x in longF]

xM = [x[0] for x in longM]

yM = [x[1] for x in longM]

fig=plt.figure(figsize=(5,5))

plt.plot(xF,yF, color="red", label="Female Names") #plot female name lengths in pink

plt.plot(xM,yM, color="blue", label="Male Names") #plot male name lengths in blue

plt.legend(loc='lower right',fontsize=12)

ax=fig.add_subplot(111)

ax.set_xlabel('Year', fontsize=12)

ax.set_ylabel('Maximum length of names')

plt.show()
cur.execute("SELECT year, SUM(count) FROM NationalNames WHERE gender='F' GROUP BY year") #total number of females by year

totalF=cur.fetchall()

cur.execute("SELECT year, SUM(count) FROM NationalNames WHERE gender='M' GROUP BY year") #total number of males by year

totalM=cur.fetchall()

xF = [x[0] for x in totalF]

yF = [x[1] for x in totalF]

fig=plt.figure(figsize=(5,5))

xM = [x[0] for x in totalM]

yM = [x[1] for x in totalM]

plt.plot(xF,yF, color="red", label="Females") #plot female name lengths in pink

plt.plot(xM,yM, color="blue", label="Males") #plot male name lengths in blue

plt.legend(loc='lower right',fontsize=12)

ax=fig.add_subplot(111)

ax.set_xlabel('Year', fontsize=12)

ax.set_ylabel('Total Population')

plt.show()
cur.execute("SELECT year, COUNT(DISTINCT name) FROM NationalNames WHERE gender='F' GROUP BY year;") #number of distinct female names grouped by year

countF=cur.fetchall()

cur.execute("SELECT year, COUNT(DISTINCT name) FROM NationalNames WHERE gender='M' GROUP BY year;") #number of distinct female names grouped by year

countM=cur.fetchall()

xF = [x[0] for x in countF]

yF = [x[1] for x in countF]

fig=plt.figure(figsize=(5,5))

xM = [x[0] for x in countM]

yM = [x[1] for x in countM]

plt.plot(xF,yF, color="red", label="Female Names") #plot female in pink

plt.plot(xM,yM, color="blue", label="Male Names") #plot male in blue

plt.legend(loc='upper left',fontsize=12)

ax=fig.add_subplot(111)

ax.set_xlabel('Year', fontsize=12)

ax.set_ylabel('Number of Distinct Names')

plt.show()
cur.execute("SELECT year, MIN(LENGTH(name)) FROM NationalNames WHERE gender='F' GROUP BY year;") #min length of female name by year

shortF=cur.fetchall()

cur.execute("SELECT year, MIN(LENGTH(name)) FROM NationalNames WHERE gender='M' GROUP BY year;") #min length of male name by year

shortM=cur.fetchall()

xF = [x[0] for x in shortF]

yF = [x[1] for x in shortF]

fig=plt.figure(figsize=(5,5))

xM = [x[0] for x in shortM]

yM = [x[1] for x in shortM]

plt.plot(xF,yF, color="red", label="Female Names") #plot female name lengths in pink

plt.plot(xM,yM, color="blue", label="Male Names") #plot male name lengths in blue

plt.legend(loc='upper right',fontsize=12)

ax=fig.add_subplot(111)

ax.set_xlabel('Year', fontsize=12)

ax.set_ylabel('Minimum Name Length')

plt.show()
cur.execute("SELECT DISTINCT name, MIN(LENGTH(name)) FROM NationalNames WHERE gender='M' GROUP BY year;")

cur.fetchall()
cur.execute("SELECT DISTINCT name, MIN(LENGTH(name)) FROM NationalNames WHERE gender='F' GROUP BY year;")

cur.fetchall()
cur.execute("SELECT year, SUM(count) FROM NationalNames WHERE gender='F' AND LENGTH(name) BETWEEN 1 AND 5 GROUP BY year;")

one2five=cur.fetchall() #stores the number of females whose name length is 1-5.

cur.execute("SELECT year, SUM(count) FROM NationalNames WHERE gender='F' AND LENGTH(name) BETWEEN 5 AND 10 GROUP BY year;")

five2ten=cur.fetchall() #stores the number of females whose name length is 5-10

cur.execute("SELECT year, SUM(count) FROM NationalNames WHERE gender='F' AND LENGTH(name) BETWEEN 10 AND 15 GROUP BY year;")

ten2fifteen=cur.fetchall() #stores the number of females whose name length is 10-15.

xshort = [x[0] for x in one2five] #these lines store the x (year) and y (total number of females) for each category.

yshort = [x[1] for x in one2five]

xmid = [x[0] for x in five2ten]

ymid = [x[1] for x in five2ten]

xlong = [x[0] for x in ten2fifteen]

ylong = [x[1] for x in ten2fifteen]

fig=plt.figure(figsize=(10,5))

plt.plot(xshort,yshort, color="red", label="Name Length 1-5 char.") #plot in red, number of females with names of length 1 to 5 characters

plt.plot(xmid,ymid, color="blue", label="Name Length 5-10 char.") #plot in blue, number of females with names of length 5 to 10 characters

plt.plot(xlong, ylong, color="green", label="Name Length 10-15 char.") #plot in green, number of females with names of length 10 to 15 characters

#plt.yscale("log")

plt.legend(loc='upper left',fontsize=12)

ax=fig.add_subplot(111)

ax.set_xlabel('Year', fontsize=12)

ax.set_ylabel('Number of Females')

plt.show()
cur.execute("SELECT name, year, count FROM NationalNames WHERE name='Lillian' AND gender='F'"+ 

            " UNION " +

            " SELECT name, year, count FROM NationalNames WHERE name='James' AND gender='M';")

landj=cur.fetchall() #returns the year and count for the names James and Lillian



#print(landj[0])

xj=[]

yj=[]

xl=[]

yl=[]

for i in range(len(landj)): #this will separate out the two names from the SQL returned data

    if landj[i][0]=='James':

            xj.append(landj[i][1]) #James

            yj.append(landj[i][2])

    else:

            xl.append(landj[i][1]) #Lillian

            yl.append(landj[i][2])

fig=plt.figure(figsize=(10,5))

plt.plot(xj,yj, color="blue", label="Name=James and Gender=M")

plt.plot(xl, yl, color="red", label="Name=Lillian and Gender=F")

plt.legend(loc='upper right',fontsize=12)

ax=fig.add_subplot(111)

ax.set_xlabel('Year', fontsize=12)

ax.set_ylabel('Number of People')

plt.show()
bryansnames=['James', 'Matthew', 'Richard'] #a list of names to iterate over

fig=plt.figure(figsize=(10,5))

for name in bryansnames:

    cur.execute("SELECT name, year, count FROM NationalNames WHERE name='" + name + "' AND gender='M';")

    output=cur.fetchall() #this will store the query for each name as the code loops over this for statement.

#    print(name)

    x=[] #creates empty lists to append to

    y=[]

    for i in range(len(output)):

        x.append(output[i][1]) #appends the year values

        y.append(output[i][2]) #appends the count values

    plt.plot(x,y,label=output[0][0]) #create a plot and label for each iternation of the for loop



#fig=plt.figure(figsize=(10,5))

plt.legend(loc='upper right',fontsize=12)

ax=fig.add_subplot(111)

ax.set_xlabel('Year', fontsize=12)

ax.set_ylabel('Number of Males')

plt.show()
cur.execute("SELECT name FROM NationalNames WHERE year='2014' AND gender='F' LIMIT 10") #returns the top ten female names for 2014.

topten=cur.fetchall() #stores the top ten most popular female names from 2014. 

toptenlist = [x[0] for x in topten] #converts this to a list of names

fig=plt.figure(figsize=(10,5))

for name in toptenlist:

    cur.execute("SELECT name, year, count FROM NationalNames WHERE name='" + name + "' AND gender='F';")

    output=cur.fetchall() #the list of names is iternated over in this for loop and stores the year and count for each name in the toptenlist.

#    print(name)

    x=[] #creates an empty list

    y=[] #creates an empty list

    for i in range(len(output)):

        x.append(output[i][1]) #appends the year values from the SQL output

        y.append(output[i][2]) #appends the count values from the SQL output

    plt.plot(x,y,label=output[0][0]) #creates of plot (for each name) and labels it.



#fig=plt.figure(figsize=(15,10))

plt.legend(loc='upper left',fontsize=12)

ax=fig.add_subplot(111)

ax.set_xlabel('Year', fontsize=12)

ax.set_ylabel('Number of Females')

plt.show()
cur.execute("SELECT name FROM NationalNames WHERE year='2014' AND gender='M' LIMIT 10") #returns the top ten males names for 2014.

topten=cur.fetchall()

toptenlist = [x[0] for x in topten]

fig=plt.figure(figsize=(10,5))

for name in toptenlist:

    cur.execute("SELECT name, year, count FROM NationalNames WHERE name='" + name + "' AND gender='M';")

    output=cur.fetchall()

#    print(name)

    x=[]

    y=[]

    for i in range(len(output)):

        x.append(output[i][1])

        y.append(output[i][2])

    plt.plot(x,y,label=output[0][0])



#fig=plt.figure(figsize=(15,10))

plt.legend(loc='upper left',fontsize=12)

ax=fig.add_subplot(111)

ax.set_xlabel('Year', fontsize=12)

ax.set_ylabel('Number of Males')

plt.show()