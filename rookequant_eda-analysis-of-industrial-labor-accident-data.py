# Importing Libraries...



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sbn

import datetime as dt

%matplotlib inline
data = pd.read_csv('../input/industrial-safety-and-health-analytics-database/IHMStefanini_industrial_safety_and_health_database_with_accidents_description.csv')
data.head(5)
data.info()
data.isnull().sum()
del data['Unnamed: 0']   # deleting unnecessary data
df = pd.DataFrame(data)
# renaming some column names



df = df.rename({'Data': 'Date', 'Genre':'Gender','Employee or Third Party':'Employee type'}, axis = 1)
# whenever I work with dataset I create "id" for each so i can extract anything with indexing

# creating ID



id =[]

for i in range(len(df)):

    id.append(i)

df['id'] = id



# Extracting Year from the date in dataset

df['Date'] = pd.to_datetime(df['Date'])  # converting into python time from string time

df['Year'] = 0        

for a,b in zip(df['Date'], df['id']):

    df['Year'][b] = a.year     # extracting year
# uniques from countries



count = []

k = 0

for i in (df['Countries']):

    if (k == 0):

        k+=1

        count.append(i)

    elif (k > 0):

        if i in count:

            k+=1

        else:

            count.append(i)

            k+=1



# unique from Locals



loc = []

k = 0

for i in (df['Local']):

    if (k == 0):

        k+=1

        loc.append(i)

    elif (k > 0):

        if i in loc:

            k+=1

        else:

            loc.append(i)

            k+=1

            

# unique from Industry sector



ins = []

k = 0

for i in (df['Industry Sector']):

    if (k == 0):

        k+=1

        ins.append(i)

    elif (k > 0):

        if i in ins:

            k+=1

        else:

            ins.append(i)

            k+=1



# unique from potrntial accident level

#(the values of risk level are already given in problem statement)



pal = ['I','II','III','IV','V','VI']

            

# unique from gender



gen = []

k = 0

for i in (df['Gender']):

    if (k == 0):

        k+=1

        gen.append(i)

    elif (k > 0):

        if i in gen:

            k+=1

        else:

            gen.append(i)

            k+=1

            

# unique from employment type



ept = []

k = 0

for i in (df['Employee type']):

    if (k == 0):

        k+=1

        ept.append(i)

    elif (k > 0):

        if i in ept:

            k+=1

        else:

            ept.append(i)

            k+=1

            

# getting the unique values from dataset and cross verifying it to check if the algorithm was correct.



print("The accidents in various countries are = {}".format(count))

print("The accidents in various locations are = {}".format(loc))

print("The accidents occured in these industrial sesctors = {}".format(ins))

print("The potential accidents risk levels are = {}".format(pal))

print("The accidents happend for = {}".format(gen))

print("The accidents happend for these employment type = {}".format(ept))

# collecting year in list to work with it.



y_r = df['Date'].dt.year

y_16 = len(y_r[y_r==2016])   # checking if there is 2016 in list and getting total numbers of it.

y_17 = len(y_r[y_r==2017])   # checking if there is 2017 in list and getting total numbers of it.

x_axis = ["2016", "2017"]

y_axis = [y_16,  y_17]

plt.bar(x_axis, y_axis, color = ['green', 'blue'])

plt.title("Accidents in 2016 and 2017")

plt.show()
# extracting country to check accidents...



countries = []

for x, y in zip(df['Countries'], df['Date']):

    if(y.year == 2016):

        countries.append(x)

        

# function to calculate number of required countries entries from the dataset 



def sum_ca(n):

    e = 0

    for c in countries:

        if ( c == n):

            e+=1

    return e



# calling function to retrieve numbers



sum_country01 = sum_ca(n = 'Country_01')

sum_country02 = sum_ca(n = 'Country_02')

sum_country03 = sum_ca(n = 'Country_03')



plt.bar(['Country_01', 'Country_02', 'Country_03'],[sum_country01, sum_country02, sum_country03])

plt.title('Accidents in 2016 ')

plt.show()
# checking types of jobs in country_01 during 2016



jobs = []

for p,q,s in zip(df['Industry Sector'], df['Date'], df['Countries']):

    if ((q.year == 2016) and (s == 'Country_01')):

        jobs.append(p)



job_MI = 0

job_MT = 0

job_OT = 0

for r in jobs:

    if(r == 'Mining'):

        job_MI+=1

    elif(r == 'Metals'):

        job_MT+=1

    elif(r == 'Others'):

        job_OT+=1



plt.bar(['Mining', 'Metals', 'Others'], [job_MI, job_MT, job_OT])

plt.title("Country_01 Accident Job report in 2016")

plt.show()
# clear visualization for checking if mining job was riskier than metals job



# mining

p = 0

q = 0

r = 0

s = 0

t = 0

u = 0



# metals

l = 0

m = 0

n = 0

o = 0

w = 0

x = 0



for a,b,c in zip(df['Accident Level'], df['Industry Sector'], df['Date'].dt.year):

    if ((a == 'I') and (b == 'Mining') and (c == 2016)):

        p+=1

    elif ((a == 'II') and (b == 'Mining') and (c == 2016)):

        q+=1

    elif ((a == 'III') and (b == 'Mining') and (c == 2016)):

        r+=1

    elif((a == 'IV') and (b == 'Mining') and (c == 2016)):

        s+=1

    elif ((a == 'V') and (b == 'Mining') and (c == 2016)):

        t+=1

    elif ((a == 'VI') and (n == 'Mining') and (c == 2016)):

        u+=1

    elif ((a == 'I') and (b == 'Metals') and (c == 2016)):

        l+=1          

    elif ((a == 'II') and (b == 'Metals') and (c == 2016)):

        m+=1

    elif ((a == 'III') and (b == 'Metals') and (c == 2016)):

        n+=1

    elif((a == 'IV') and (b == 'Metals') and (c == 2016)):

        o+=1

    elif ((a == 'V') and (b == 'Metals') and (c == 2016)):

        w+=1

    elif ((a == 'VI') and (n == 'Metals') and (c == 2016)):

        x+=1



# combining two bars to visulize if mining level of risk was more than metals.



plt.bar(['I','II','III','IV','V','VI'], [p,q,r,s,t,u], color = 'green', label = 'Mining')

plt.bar(['I','II','III','IV','V','VI'], [l,m,n,o,w,x], color = 'blue', label = 'Metals')

plt.title('Mining and Metals Levels')

plt.legend()

plt.show()

# clear visualization for heighest mining job in each country of 2016



# country1

p = 0

q = 0

r = 0

# country 2

l = 0

m = 0

n = 0

# country 3

o = 0

s = 0

t = 0

for a,b,c in zip(df['Countries'], df['Industry Sector'], df['Date'].dt.year):

    if ((a == 'Country_01') and (b == 'Mining') and (c == 2016)):

        p+=1

    elif ((a == 'Country_01') and (b == 'Metals') and (c == 2016)):

        q+=1

    elif ((a == 'Country_01') and (b == 'Others') and (c == 2016)):

        r+=1

    elif((a == 'Country_02') and (b == 'Mining') and (c == 2016)):

        l+=1

    elif ((a == 'Country_02') and (b == 'Metals') and (c == 2016)):

        m+=1

    elif ((a == 'Country_02') and (n == 'Others') and (c == 2016)):

        n+=1

    elif((a == 'Country_03') and (b == 'Mining') and (c == 2016)):

        o+=1

    elif ((a == 'Country_03') and (b == 'Metals') and (c == 2016)):

        s+=1

    elif ((a == 'Country_03') and (n == 'Others') and (c == 2016)):

        t+=1



mining_C1 = p

metals_C1 = q

others_C1 = r



mining_C2 = l

metals_C2 = m

others_C2 = n



mining_C3 = o

metals_C3 = s

others_C3 = t



C1 = [mining_C1, metals_C1, others_C1]

C2 = [mining_C2, metals_C2, others_C2]

C3 = [mining_C3, metals_C3, others_C3]



# position



x_width = 0.25

pos1 = np.arange(len(C1))

pos2 = [ax + x_width for ax in pos1]

pos3 = [ax + x_width for ax in pos2]



plt.bar(pos1, C1, color = 'green', width = x_width, label = 'mining')

plt.bar(pos2, C2, color = 'red', width = x_width, label = 'metals')

plt.bar(pos3, C3, color = 'blue', width = x_width, label = 'others')



plt.xlabel('countries')

plt.xticks([pos + x_width for pos in range(len(C1))], ['country_01', 'country_02', 'country_03'])

plt.legend()

plt.show()

# Accidents in each country



countries_17 = []

for x, y in zip(df['Countries'], df['Date']):

    if(y.year == 2017):

        countries_17.append(x)

def sum_ac(n):

    e = 0

    for c in countries_17:

        if ( c == n):

            e+=1

    return e



sum_country_2017_01 = sum_ac(n = 'Country_01')

sum_country_2017_02 = sum_ac(n = 'Country_02')

sum_country_2017_03 = sum_ac(n = 'Country_03')



plt.bar(['Country_01', 'Country_02', 'Country_03'],[sum_country_2017_01, sum_country_2017_02, sum_country_2017_03])

plt.title('Accidents in 2017')

plt.show()
# analysing jobs in country_01



jobs_17 = []

for p,q,s in zip(df['Industry Sector'], df['Date'], df['Countries']):

    if ((q.year == 2017) and (s == 'Country_01')):

        jobs_17.append(p)



job_MI = 0

job_MT = 0

job_OT = 0

for r in jobs_17:

    if(r == 'Mining'):

        job_MI+=1

    elif(r == 'Metals'):

        job_MT+=1

    elif(r == 'Others'):

        job_OT+=1



plt.bar(['Mining', 'Metals', 'Others'], [job_MI, job_MT, job_OT])

plt.title("Country_01 Accident Job report in 2017")

plt.show()
# clear visualization for checking if mining is riskier than metals job in 2017



# mining

p = 0

q = 0

r = 0

s = 0

t = 0

u = 0



# metals

l = 0

m = 0

n = 0

o = 0

w = 0

x = 0



for a,b,c in zip(df['Accident Level'], df['Industry Sector'], df['Date'].dt.year):

    if ((a == 'I') and (b == 'Mining') and (c == 2017)):

        p+=1

    elif ((a == 'II') and (b == 'Mining') and (c == 2017)):

        q+=1

    elif ((a == 'III') and (b == 'Mining') and (c == 2017)):

        r+=1

    elif((a == 'IV') and (b == 'Mining') and (c == 2017)):

        s+=1

    elif ((a == 'V') and (b == 'Mining') and (c == 2017)):

        t+=1

    elif ((a == 'VI') and (n == 'Mining') and (c == 2017)):

        u+=1

    elif ((a == 'I') and (b == 'Metals') and (c == 2017)):

        l+=1          

    elif ((a == 'II') and (b == 'Metals') and (c == 2017)):

        m+=1

    elif ((a == 'III') and (b == 'Metals') and (c == 2017)):

        n+=1

    elif((a == 'IV') and (b == 'Metals') and (c == 2017)):

        o+=1

    elif ((a == 'V') and (b == 'Metals') and (c == 2017)):

        w+=1

    elif ((a == 'VI') and (n == 'Metals') and (c == 2017)):

        x+=1





plt.bar(['I','II','III','IV','V','VI'], [p,q,r,s,t,u], color = 'green', label = 'Mining')

plt.bar(['I','II','III','IV','V','VI'], [l,m,n,o,w,x], color = 'blue', label = 'Metals')

plt.title('Mining and Metals Levels of 2017')

plt.legend()

plt.show()
# total number of mining jobs in 2016 and 2017



y = 0

z = 0

for p,q in zip(df['Industry Sector'], df['Year']):

    if ((p == 'Mining') and ( q == 2016)):

        y+=1

    elif((p == 'Mining') and (q == 2017)):

        z+=1



mining_16 = y 

mining_17 = z

x_val = ['2016', '2017']

y_val = [mining_16, mining_17] 



ax = sbn.pointplot(x = x_val, y = y_val )