import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

data_file = pd.read_csv('../input/2016-FCC-New-Coders-Survey-Data.csv', dtype={'AttendedBootcamp': float, 'CodeEventOther': object, 'JobRoleInterestOther': object})

AttendedBootcamp = data_file['AttendedBootcamp']

BootcampFullJob = data_file['BootcampFullJobAfter']

BootcampRecommend = data_file['BootcampRecommend']

BootcampLoan = data_file['BootcampLoanYesNo']

BootcampName = data_file['BootcampName']

FinanciallySupporting = data_file['FinanciallySupporting']

HasChildren = data_file['HasChildren']

HasDebt = data_file['HasDebt']

FinancialDependents = data_file['HasFinancialDependents']

EmploymentStatus = data_file['EmploymentStatus']

EmploymentStatusOther = data_file['EmploymentStatusOther']

BootcampFinish = data_file['BootcampFinish']

Age = data_file['Age']

NetworkID = data_file['NetworkID']



AttendYes = data_file[data_file.AttendedBootcamp == 1]

AgeAttend = AttendYes['Age']

FinishYes = data_file[data_file.BootcampFinish == 1]

FinishNo = data_file[data_file.BootcampFinish == 0]

JobYes = data_file[data_file.BootcampFullJobAfter == 1]

JobNo = data_file[data_file.BootcampFullJobAfter == 0]

RecYes = data_file[data_file.BootcampRecommend == 1]

RecNo = data_file[data_file.BootcampRecommend == 0]

"""is there a way to combine attributes using the variable names?"""

RecYesJobYes = data_file[data_file.BootcampRecommend == 1][data_file.BootcampFullJobAfter == 1 ]

RecNoJobYes = data_file[data_file.BootcampRecommend == 0][data_file.BootcampFullJobAfter == 1 ]

RecYesJobNo = data_file[data_file.BootcampRecommend == 1][data_file.BootcampFullJobAfter == 0 ]

RecNoJobNo = data_file[data_file.BootcampRecommend == 0][data_file.BootcampFullJobAfter == 0 ]
plt.figure(figsize=(8,8))

plt.title('Attend bootcamp')

plt.hist(AttendYes['Age'], histtype='bar', range=[16,61])

plt.xlabel('Age')

plt.ylabel('Count')

plt.legend()

plt.show()
numjobfin2 = [len(JobNo[JobNo.Age == i]) for i in range(16, 61)]

numjobfin1 = [len(JobYes[JobYes.Age == i]) for i in range(16, 61)]

demjobfin = [len(JobYes[FinishYes.Age == i]) + len(JobNo[JobNo.Age == i]) for i in range(16, 61)]



Yesjonfin =[int(m) / int(b) if int(b) != 0 else int(m) for b,m in zip(demjobfin, numjobfin1)]

Nojobfin = [int(m) / int(b) if int(b) != 0 else int(m) for b,m in zip(demjobfin, numjobfin2)]



plt.figure(figsize=(8,8))

plt.title('bootcamp job normed')

plt.xlabel('Age')

plt.ylabel('Count normed')

plt.bar(np.arange(16, 61, 1), Yesjonfin, 0.35, label='job')

plt.bar(np.arange(16, 61, 1) + 0.35, Nojobfin, 0.35, label='no job')

plt.xticks(np.arange(15, 65, 5))

plt.legend()

plt.show()





nurator = [len(JobYes[JobYes.Age == i]) - len(JobNo[JobNo.Age == i]) for i in range(16, 60)]

x = range(16, 60)  

derator = [len(JobYes[JobYes.Age == i]) + len(JobNo[JobNo.Age == i]) for i in range(16, 60)]

bananasplit = []

for b, m in zip(derator, nurator):

    try:

        bananasplit.append(int(m)/int(b))

    except ZeroDivisionError:

        bananasplit.append(int(m))

fig = plt.figure(figsize=(8,8))

plt.plot(x, bananasplit, 'go')

plt.xlabel('Age')

plt.ylabel('Net Employment Difference (count)')

plt.title('Employement Discrepencies normed')

plt.xticks(x)

plt.xscale('linear')

ax = fig.add_subplot(1, 1, 1)

ax.spines['bottom'].set_position('zero')

plt.vlines(x, [0], bananasplit)           

ax.xaxis.set_ticks(np.arange(15, 65, 5))

plt.xlabel('Age', horizontalalignment='center', verticalalignment='center', x=1.05)

plt.show()







numerator1 =[len(RecYes[RecYes.Age == i]) - len(RecNo[RecNo.Age == i])for i in range(16, 61)]

denomerator1 = [len(RecYes[RecYes.Age == i]) + len(RecNo[RecNo.Age == i])for i in range(16, 61)]

numerator2 =[len(RecYesJobYes[RecYesJobYes.Age == i]) - len(RecNoJobYes[RecNoJobYes.Age == i]) for i in range(16, 61)]

denomerator2 = [len(RecYesJobYes[RecYesJobYes.Age == i]) + len(RecNoJobYes[RecNoJobYes.Age == i]) for i in range(16, 61)]

numerator3 = [len(RecYesJobNo[RecYesJobNo.Age == i]) - len(RecNoJobNo[RecNoJobNo.Age == i]) for i in range(16, 61)]

denomerator3 = [len(RecYesJobNo[RecYesJobNo.Age == i]) + len(RecNoJobNo[RecNoJobNo.Age == i]) for i in range(16, 61)]



d1 = {'numerator1': numerator1, 'denomerator1': denomerator1}

d2 = {'numerator2': numerator2, 'denomerator2': denomerator2}

d3 = {'numerator3': numerator3, 'denomerator3': denomerator3}



df1 = pd.DataFrame(data=d1)

df2 = pd.DataFrame(data=d2)

df3 = pd.DataFrame(data=d3)



y1 = df1['result'] = df1.numerator1.div(df1.denomerator1)

df1.loc[~np.isfinite(df1['result']), 'result'] = np.nan

        

y2 = df2['result'] = df2.numerator2.div(df2.denomerator2)

df2.loc[~np.isfinite(df2['result']), 'result'] = np.nan   



y3 = df3['result'] = df3.numerator3.div(df3.denomerator3)

df3.loc[~np.isfinite(df3['result']), 'result'] = np.nan



x = range(16, 61) 

fig = plt.figure(figsize=(8,8))

plt.plot(x, y1, label = 'all')

plt.plot(x, y2, label = 'job afterwards')

plt.plot(x, y3, label = 'no job afterwards')

plt.xlabel('Age')

plt.ylabel('Net reccomendation (count)')

plt.title('Age Sentiment')

plt.xticks(x)

plt.xscale('linear')

ax = fig.add_subplot(1, 1, 1)         

ax.xaxis.set_ticks(np.arange(15, 65, 5))

plt.xlabel('Age', horizontalalignment='center', verticalalignment='center')

plt.legend()

plt.show()
numfinish2 = [len(FinishNo[FinishNo.Age == i]) for i in range(16, 61)]

numfinish1 = [len(FinishYes[FinishYes.Age == i]) for i in range(16, 61)]

demfinish = [len(FinishYes[FinishYes.Age == i]) + len(FinishNo[FinishNo.Age == i]) for i in range(16, 61)]



YesFinish =[int(m) / int(b) if int(b) != 0 else int(m) for b,m in zip(demfinish, numfinish1)]

NoFinish = [int(m) / int(b) if int(b) != 0 else int(m) for b,m in zip(demfinish, numfinish2)]



plt.figure(figsize=(8,8))

plt.title('bootcampfinish normed')

plt.xlabel('Age')

plt.ylabel('Count')

plt.bar(np.arange(16, 61, 1), YesFinish, 0.35, label='finished')

plt.bar(np.arange(16, 61, 1) + 0.35, NoFinish, 0.35, label='didnt finish')

plt.xticks(np.arange(15, 65, 5))

plt.legend()

plt.show()
import csv, sqlite3

import pandas as pd



con = sqlite3.connect("database_survey.db")

cur=con.cursor() # Get the cursor

csv_data = csv.reader('../input/2016-FCC-New-Coders-Survey-Data.csv')



with open('../input/2016-FCC-New-Coders-Survey-Data.csv', 'r', encoding="utf8") as fin: # `with` statement available in 2.5+

    # csv.DictReader uses first line in file for column headings by default

    dr = csv.DictReader(fin) # comma is default delimiter

    to_db = [(i['BootcampRecommend'], i['BootcampFullJobAfter']) for i in dr]

    

with con:

    cur = con.cursor()

    cur.execute("DROP TABLE IF EXISTS MyTable")

    cur.execute("CREATE TABLE MyTable(BootcampRecommend, BootcampFullJobAfter);") # use your column names here

    for rows in to_db:

        cur.execute('INSERT INTO MyTable VALUES(?,?)', rows)

    cur.fetchall()

    cur.execute("SELECT COUNT(BootcampRecommend), BootcampFullJobAfter FROM MyTable WHERE BootcampRecommend='1' GROUP BY BootcampFullJobAfter")

    rows = cur.fetchall()

    

    cur.execute("SELECT COUNT(*) FROM MyTable where BootcampFullJobAfter = '1'")

    rows2 = cur.fetchall()

    

    cur.execute("SELECT COUNT(BootcampRecommend), BootcampFullJobAfter FROM MyTable WHERE BootcampRecommend='0' GROUP BY BootcampFullJobAfter")

    rows3 = cur.fetchall()

    

con.commit()

con.close()    



import numpy as np

import matplotlib.pyplot as plt



values, labels = zip(*rows[0:2]) #assigns the individual elements

values3, labels3 = zip(*rows3[0:2])

"""the unpacking operator * breaks up the list of lists into just two lists so 

when the zip() function is applied, it merges the values of the first element 

in each list """

xlocs = np.array([0, 1])

plt.bar(xlocs - 0.2, values,width=0.2,color='b',align='center', label='reccomendation')

plt.bar(xlocs, values3,width=0.2,color='g',align='center', label='no reccomendation')

plt.title("Getting Job and Reccomendation")

plt.xticks(xlocs - 0.1,  ["no job", "job"])

plt.legend(loc=(1.03,0.2))

plt.show()