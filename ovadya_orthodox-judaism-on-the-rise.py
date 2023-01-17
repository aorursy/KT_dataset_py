# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from collections import OrderedDict
import seaborn as sns


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
world = pd.read_csv('../input/global.csv')
region = pd.read_csv('../input/regional.csv')
national = pd.read_csv('../input/national.csv')

# Any results you write to the current directory are saved as output.
list(national['year'].unique())
fig, axis = plt.subplots(figsize=(20,10))
#axis.yaxis.grid(True)
#axis.xaxis.grid(True)

axis.set_xlim(1945, 2010)
axis.set_ylim(0,7000000)
plt.plot(world['year'].values, world['judaism_orthodox'].values)

axis.set_title('Rise in followers of orthodox Judaism since 1945',fontsize=25)
axis.set_xlabel('year',fontsize=20)
axis.set_ylabel('number of orthodox Jews',fontsize=20)
Y = world['year'].values
X1 = world['judaism_orthodox'].values
axis.fill_between(Y, 0, X1,facecolor='black', alpha=0.9)

line_X1 = axis.plot(Y, X1, label = "orthodox",color='black')

fig, axis = plt.subplots(figsize=(20,10))
# Grid lines, Xticks, Xlabel, Ylabel

axis.yaxis.grid(True)
axis.xaxis.grid(True)

axis.set_title('Number of followers of the different streams of Judaism',fontsize=25)
axis.set_xlabel('year',fontsize=20)
axis.set_ylabel('number of followers',fontsize=20)
axis.set_xlim(1945, 2010)

Y = world['year']
X1 = world['judaism_orthodox']
X2 = world['judaism_conservative']
X3 = world['judaism_reform']
X4 = world['judaism_other']
X5 = world['judaism_all']

line_X1 = axis.plot(Y, X1, label = "orthodox", linewidth=10, linestyle="-", c="black")
line_X2 = axis.plot(Y, X2, label = "conservative", linewidth=10, linestyle="-", c="blue")
line_X3 = axis.plot(Y, X3, label = "reform",linewidth=10, linestyle="-", c="green")
line_X4 = axis.plot(Y, X4, label = "other",linewidth=10, linestyle="-", c="orange")
line_X5 = axis.plot(Y, X5,linewidth =10, linestyle="-", c="purple")


plt.legend(bbox_to_anchor=(1.05, 1), loc=2,
           ncol=1,prop={'size': 18}, borderaxespad=0.)
plt.show()
fig, axis = plt.subplots(figsize=(20,10))
# Grid lines, Xticks, Xlabel, Ylabel

axis.yaxis.grid(True)
axis.xaxis.grid(True)

axis.set_xlabel('year',fontsize=20)
axis.set_ylabel('percentage of followers',fontsize=20)
axis.set_title('Proportional distribution of the different streams in Judaism',fontsize=25)


Y = world['year']
X1 = world['judaism_orthodox']/ world['judaism_all']
X2 = world['judaism_conservative']/ world['judaism_all']
X3 = world['judaism_reform']/ world['judaism_all']
X4 = world['judaism_other']/ world['judaism_all']
X5 = world['judaism_all']/ world['judaism_all']

line_X1 = axis.plot(Y, X1, label = "orthodox", linewidth=10, linestyle="-", c="black",
         solid_capstyle="round")
line_X2 = axis.plot(Y, X2, label = "conservative", linewidth=10, linestyle="-", c="blue",
         solid_capstyle="round")
line_X3 = axis.plot(Y, X3, label = "reform", linewidth=10, linestyle="-", c="green",
         solid_capstyle="round")
line_X4 = axis.plot(Y, X4, label = "other", linewidth=10, linestyle="-", c="orange",
         solid_capstyle="round")
line_X5 = axis.plot(Y, X5, label ="all", linewidth=5, linestyle="-", c="purple",
         solid_capstyle="round")


plt.legend(bbox_to_anchor=(1.05, 1), loc=2,
           ncol=1,prop={'size': 18}, borderaxespad=0.)
plt.show()
print(world['judaism_all'].tail(1))
print((14.310-14.023)/14.310)
fig, axis = plt.subplots(figsize=(20,10))
# Grid lines, Xticks, Xlabel, Ylabel

axis.yaxis.grid(True)
axis.xaxis.grid(True)
axis.set_xlim(1945, 2010)
axis.set_ylim(0,1)

axis.set_title('Percentual distribution of the different streams in Judaism',fontsize=25)
axis.set_xlabel('year',fontsize=20)
axis.set_ylabel('percentage of followers',fontsize=20)

#line_ort = axis.plot(year, ort, label = "real orthodox")

Y = world['year']
X1 = world['judaism_orthodox']/ world['judaism_all']
X2 = world['judaism_conservative']/ world['judaism_all']
X3 = world['judaism_reform']/ world['judaism_all']
X4 = world['judaism_other']/ world['judaism_all']
X5 = world['judaism_all']/ world['judaism_all']
axis.stackplot(Y.values.flatten('F'), X1.values.flatten('F'), X2.values.flatten('F'), X3.values.flatten('F'), X4.values.flatten('F'),colors=['black','blue','green','orange'],labels=['orthodox','conservatice','reform','other'])

plt.legend(bbox_to_anchor=(1.05, 1), loc=2,
           ncol=1,prop={'size': 18}, borderaxespad=0.)
plt.show()
labels = 'orthodox', 'conservative', 'reform', 'other'
size1945 = [X1[0], X2[0], X3[0], X4[0]]
size1975 = [X1[6],X2[6],X3[6],X4[6]]
size2010 = [X1[13], X2[13],X3[13],X4[13]]
explode = (0, 0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=[30,10])

ax1.set_title('1945',fontsize=25)
ax1.pie(size1945, explode=explode, labels=labels, textprops={'fontsize':18},
        shadow=True, startangle=90,colors=['black','blue','green','orange'])
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

ax2.set_title('1975',fontsize=25)
ax2.pie(size1975, explode=explode, labels=labels, textprops={'fontsize':18},
        shadow=True, startangle=90,colors=['black','blue','green','orange'])
ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

ax3.set_title('2010',fontsize=25)
ax3.pie(size2010, explode=explode, labels=labels,textprops={'fontsize':18},
        shadow=True, startangle=90,colors=['black','blue','green','orange'])
ax3.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.suptitle('Proportional distribution of followers of different streams of Judaism at selected timepoints ',fontsize=30)
plt.show()
Y = world['year']
X1 = world['judaism_orthodox']
X2 = world['judaism_conservative']
X3 = world['judaism_reform']
X4 = world['judaism_other']
X5 = world['judaism_all']

year = Y.values.reshape(-1,1)
ort = X1.values.reshape(-1,1)
con = X2.values.reshape(-1,1)
ref = X3.values.reshape(-1,1)
oth = X4.values.reshape(-1,1)
together = X5.values.reshape(-1,1)
ort_train, ort_test, y_train, y_test = train_test_split(ort, year, test_size=0.25, random_state=665)
ort_regressor = LinearRegression()
ort_regressor.fit(y_train,ort_train)
ort_prediction = ort_regressor.predict(y_test)
RMSE = sqrt(mean_squared_error(y_true = ort_test, y_pred = ort_prediction))
print("{0:.2f}".format(RMSE/(np.amax(ort_test)-np.amin(ort_test))), " normalized RMSE")
print("{0:.2f}".format(ort_test.std()/(np.amax(ort_test)-np.amin(ort_test))), " normalized STD test data")
print("{0:.2f}".format(ort.std()/(np.amax(ort)-np.amin(ort))), " normalized STD all data")

con_train, con_test, y_train, y_test = train_test_split(con, year, test_size=0.25, random_state=665)
con_regressor = LinearRegression()
con_regressor.fit(y_train,con_train)
con_prediction = con_regressor.predict(y_test)
RMSE = sqrt(mean_squared_error(y_true = con_test, y_pred = con_prediction))
print("{0:.2f}".format(RMSE/(np.amax(con_test)-np.amin(con_test))), " normalized RMSE")
print("{0:.2f}".format(con_test.std()/(np.amax(con_test)-np.amin(con_test))), " normalized STD test data")
print("{0:.2f}".format(con.std()/(np.amax(con)-np.amin(con))), " normalized STD all data")

ref_train, ref_test, y_train, y_test = train_test_split(ref, year, test_size=0.25, random_state=665)
ref_regressor = LinearRegression()
ref_regressor.fit(y_train,ref_train)
ref_prediction = ref_regressor.predict(y_test)
RMSE = sqrt(mean_squared_error(y_true = ref_test, y_pred = ref_prediction))
print("{0:.2f}".format(RMSE/(np.amax(ref_test)-np.amin(ref_test))), " normalized RMSE")
print("{0:.2f}".format(ref_test.std()/(np.amax(ref_test)-np.amin(ref_test))), " normalized STD test data")
print("{0:.2f}".format(ref.std()/(np.amax(ref)-np.amin(ref))), " normalized STD all data")

oth_train, oth_test, y_train, y_test = train_test_split(oth, year, test_size=0.25, random_state=665)
oth_regressor = LinearRegression()
oth_regressor.fit(y_train,oth_train)
oth_prediction = oth_regressor.predict(y_test)
RMSE = sqrt(mean_squared_error(y_true = oth_test, y_pred = oth_prediction))
print("{0:.2f}".format(RMSE/(np.amax(oth_test)-np.amin(oth_test))), " normalized RMSE")
print("{0:.2f}".format(oth_test.std()/(np.amax(oth_test)-np.amin(oth_test))), " normalized STD test data")
print("{0:.2f}".format(oth.std()/(np.amax(oth)-np.amin(oth))), " normalized STD all data")

fig, axis = plt.subplots(figsize=(20,10))
# Grid lines, Xticks, Xlabel, Ylabel

axis.yaxis.grid(True)
axis.xaxis.grid(True)

axis.set_title('Linear Regression Model prediction and real data of followers of Judaism',fontsize=25)
axis.set_xlabel('year',fontsize=20)
axis.set_ylabel('number of followers',fontsize=20)


line_o_p = axis.plot(year, ort_regressor.predict(year), label = "predicted orthodox", linewidth=16, linestyle="-", c="black",
         solid_capstyle="round", alpha=0.5)
line_ort = axis.plot(year, ort, label = "real orthodox", linewidth=8, linestyle="-.", c="black",
         solid_capstyle="round")


line_c_p = axis.plot(year, con_regressor.predict(year), label = "predicted conservative", linewidth=16, linestyle="-", c="green",
         solid_capstyle="round", alpha = 0.5)
line_con = axis.plot(year, con, label = "real conservative", linewidth=8, linestyle="-.", c="green",
         solid_capstyle="round")


line_r_p = axis.plot(year, ref_regressor.predict(year), label = "predicted reform", linewidth=16, linestyle="-", c="blue",
         solid_capstyle="round",alpha=0.5)
line_ref = axis.plot(year, ref, label = "real reform", linewidth=8, linestyle="-.", c="blue",
         solid_capstyle="round")


line_t_p = axis.plot(year, oth_regressor.predict(year), label = "predicted other", linewidth=16, linestyle="-", c="orange",
         solid_capstyle="round",alpha=0.5)
line_oth = axis.plot(year, oth, label = "real other", linewidth=8, linestyle="-.", c="orange",
         solid_capstyle="round")


line_a_p = axis.plot(year,np.sum(np.array([ort_regressor.predict(year),con_regressor.predict(year),ref_regressor.predict(year),oth_regressor.predict(year)]),axis=0), label ="predicted sum", linewidth=16, linestyle="-", c="purple",
         solid_capstyle="round", alpha=0.5)
line_all = axis.plot(year,together, label ="real sum", linewidth=8, linestyle="-.", c="purple",
         solid_capstyle="round")


plt.legend(bbox_to_anchor=(1.05, 1), loc=2,
           ncol=1,prop={'size': 18}, borderaxespad=0.)
plt.show()

future = np.array([1945,1950,1955,1960,1965,1970,1975,1980,1985,1990,1995,2000,2005,2010,2015,2020,2025,2030,2035,2040,2045,2050,2055,2060,2065]).reshape(-1,1)
ort_future = ort_regressor.predict(future)
con_future = con_regressor.predict(future)
ref_future = ref_regressor.predict(future)
oth_future = oth_regressor.predict(future)
array_all_future = np.array([ort_future,con_future,ref_future,oth_future])
sum_all = np.sum(array_all_future, axis = 0 )
print(array_all_future.shape)
print(sum_all.shape)
print(future.shape)

fig, axis = plt.subplots(figsize=(20,10))
# Grid lines, Xticks, Xlabel, Ylabel

axis.yaxis.grid(True)
axis.xaxis.grid(True)

axis.set_title('Prediction future numbers of followers of different stream of Judaism',fontsize=25)
axis.set_xlabel('year',fontsize=20)
axis.set_ylabel('number of followers',fontsize=20)

#line_ort = axis.plot(year, ort, label = "real orthodox")
line_o_p = axis.plot(future, ort_future, label = "predicted orthodox",linewidth=16, linestyle="-", c="black",
         solid_capstyle="round",alpha=0.5)

#line_con = axis.plot(year, con, label = "real conservative")
line_c_p = axis.plot(future, con_future, label = "predicted conservative",linewidth=16, linestyle="-", c="green",
         solid_capstyle="round",alpha=0.5)

#line_ref = axis.plot(year, ref, label = "real reform")
line_r_p = axis.plot(future, ref_future, label = "predicted reform",linewidth=16, linestyle="-", c="blue",
         solid_capstyle="round",alpha=0.5)

#line_oth = axis.plot(year, oth, label = "real other")
line_t_p = axis.plot(future, oth_future, label = "predicted other",linewidth=16, linestyle="-", c="orange",
         solid_capstyle="round",alpha=0.5)

line_t_p = axis.plot(future, sum_all, label = "predicted all",linewidth=16, linestyle="-", c="purple",
         solid_capstyle="round",alpha=0.5)


plt.legend(bbox_to_anchor=(1.05, 1), loc=2,
           ncol=1,prop={'size': 18}, borderaxespad=0.)
plt.show()
orthodox = ort_future/sum_all
conservative = con_future/sum_all
reform = ref_future/sum_all
others = oth_future/sum_all
fig, axis = plt.subplots(figsize=(20,10))
# Grid lines, Xticks, Xlabel, Ylabel

axis.yaxis.grid(True)
axis.xaxis.grid(True)

axis.set_title('Prediction of proportional distribution of the different streams in Judaism',fontsize=25)

axis.set_xlabel('year',fontsize=20)
axis.set_ylabel('percentage of followers',fontsize=20)

#line_ort = axis.plot(year, ort, label = "real orthodox")
line_o_p = axis.plot(future, orthodox, label = "predicted orthodox", linewidth=16, linestyle="-", c="black",
         solid_capstyle="round",alpha=0.5)

#line_con = axis.plot(year, con, label = "real conservative")
line_c_p = axis.plot(future, conservative, label = "predicted conservative", linewidth=16, linestyle="-", c="green",
         solid_capstyle="round",alpha=0.5)

#line_ref = axis.plot(year, ref, label = "real reform")
line_r_p = axis.plot(future, reform, label = "predicted reform", linewidth=16, linestyle="-", c="blue",
         solid_capstyle="round",alpha=0.5)

#line_oth = axis.plot(year, oth, label = "real other")
line_t_p = axis.plot(future, others, label = "predicted other", linewidth=16, linestyle="-", c="orange",
         solid_capstyle="round",alpha=0.5)

line_t_p = axis.plot(future, sum_all/sum_all, label = "predicted all", linewidth=8, linestyle="-", c="purple",
         solid_capstyle="round",alpha=0.5)



plt.legend(bbox_to_anchor=(1.05, 1), loc=2,
           ncol=1,prop={'size': 18}, borderaxespad=0.)
plt.show()
fig, axis = plt.subplots(figsize=(20,10))
# Grid lines, Xticks, Xlabel, Ylabel

axis.yaxis.grid(True)
axis.xaxis.grid(True)
axis.set_ylim(0,1)
axis.set_xlim(1945,2065)
axis.set_title('Prediction of percentual distribution of the different streams in Judaism',fontsize=25)
axis.set_xlabel('year',fontsize=20)
axis.set_ylabel('percentage of followers',fontsize=20)

#line_ort = axis.plot(year, ort, label = "real orthodox")


axis.stackplot(future.flatten(), orthodox.flatten(), conservative.flatten(), reform.flatten(), others.flatten(),colors=['black','blue','green','orange'],labels=['orthodox','conservatice','reform','other'])

plt.legend(bbox_to_anchor=(1.05, 1), loc=2,
           ncol=1,prop={'size': 18}, borderaxespad=0.)
plt.show()
(2065-1945)/8


labels = 'orthodox', 'conservative', 'reform', 'other'
size1945 =[orthodox[0],conservative[0],reform[0],others[0]]
size1960 =[orthodox[3],conservative[3],reform[3],others[3]]
size1975 =[orthodox[6],conservative[6],reform[6],others[6]]
size1990=[orthodox[9],conservative[9],reform[9],others[9]]
size2005=[orthodox[12],conservative[12],reform[12],others[12]]
size2020=[orthodox[15],conservative[15],reform[15],others[15]]
size2035=[orthodox[18],conservative[18],reform[18],others[18]]
size2050=[orthodox[21],conservative[21],reform[21],others[21]]
size2065=[orthodox[24],conservative[24],reform[24],others[24]]

explode = (0, 0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig, axes = plt.subplots(nrows=3,ncols= 3, figsize=(30,20))

axes[0, 0].set_title('1945',fontsize=25)
axes[0, 0].pie(size1945, explode=explode, labels=labels, textprops={'fontsize':18},
        shadow=True, startangle=90,colors=['black','blue','green','orange'])
axes[0, 0].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

axes[0, 1].set_title('1960',fontsize=25)
axes[0, 1].pie(size1960, explode=explode, labels=labels, textprops={'fontsize':18},
        shadow=True, startangle=90,colors=['black','blue','green','orange'])
axes[0, 1].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

axes[0, 2].set_title('1975',fontsize=25)
axes[0, 2].pie(size1975, explode=explode, labels=labels,textprops={'fontsize':18},
        shadow=True, startangle=90,colors=['black','blue','green','orange'])
axes[0, 2].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

axes[1, 0].set_title('1990',fontsize=25)
axes[1, 0].pie(size1990, explode=explode, labels=labels, textprops={'fontsize':18},
        shadow=True, startangle=90,colors=['black','blue','green','orange'])
axes[1, 0].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

axes[1, 1].set_title('2005',fontsize=25)
axes[1, 1].pie(size2005, explode=explode, labels=labels, textprops={'fontsize':18},
        shadow=True, startangle=90,colors=['black','blue','green','orange'])
axes[1,1].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

axes[1, 2].set_title('2020',fontsize=25)
axes[1, 2].pie(size2020, explode=explode, labels=labels,textprops={'fontsize':18},
        shadow=True, startangle=90,colors=['black','blue','green','orange'])
axes[1, 2].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
axes[2, 0].set_title('2035',fontsize=25)
axes[2, 0].pie(size2035, explode=explode, labels=labels, textprops={'fontsize':18},
        shadow=True, startangle=90,colors=['black','blue','green','orange'])
axes[2, 0].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

axes[2, 1].set_title('2050',fontsize=25)
axes[2, 1].pie(size2050, explode=explode, labels=labels, textprops={'fontsize':18},
        shadow=True, startangle=90,colors=['black','blue','green','orange'])
axes[2,1].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

axes[2, 2].set_title('2065',fontsize=25)
axes[2, 2].pie(size2065, explode=explode, labels=labels,textprops={'fontsize':18},
        shadow=True, startangle=90,colors=['black','blue','green','orange'])
axes[2, 2].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.suptitle('Estimated proportional distribution of followers of different streams of Judaism at selected timepoints ',fontsize=30)
plt.show()
national_jews = national[['year','state','judaism_orthodox','judaism_conservative','judaism_reform','judaism_other','judaism_all']]
national_jews['year'].unique().shape
np.sum(national_jews['judaism_all']>10000)
national_jews['state'][national_jews['judaism_all']>10000].unique()
nations = national_jews[national_jews['state'].isin(national_jews['state'][national_jews['judaism_all']>10000].unique())]
len(nations['state'].unique())
nations.shape
nations[nations['state'].str.contains('France')]
nations[(nations['judaism_conservative']>1) | (nations['judaism_orthodox']>1) | (nations['judaism_reform']>1)]
8.8*0.75
5517567/6600000
national_jews.head()
diaspora = national_jews[(national_jews['state'] != 'Israel') & (national_jews['state'] != 'United States of America') & (national_jews['state'] != 'Canada')]
del diaspora['state']
del diaspora['judaism_orthodox']
del diaspora['judaism_conservative']
del diaspora['judaism_reform']
del diaspora['judaism_other']
diaspora.head()
for y in year:
    y=int(y)

diaspora_dict ={}
a=0
for y in diaspora['year']:
    if y not in diaspora_dict:
        diaspora_dict[y] = diaspora['judaism_all'].iloc[a]
    else:
        diaspora_dict[y] += diaspora['judaism_all'].iloc[a]

    a += 1
print(diaspora_dict)
print(diaspora_dict)
whole_diaspora = pd.Series(diaspora_dict).values.reshape(-1,1)
print(whole_diaspora)
year = world['year'].values.reshape(-1,1)
ort = world['judaism_orthodox'].values.reshape(-1,1)
oth = world['judaism_other'].values.reshape(-1,1)
israeli_all = national_jews['judaism_all'][national_jews['state']=='Israel'].values.reshape(-1,1)
israeli_all.shape
fig, axis = plt.subplots(figsize=(20,10))
# Grid lines, Xticks, Xlabel, Ylabel

axis.yaxis.grid(True)
axis.xaxis.grid(True)

axis.set_title('Number of diaspora and Israeli Jews correspond with "other" and "orthodox" respectively',fontsize=25)
axis.set_xlabel('year',fontsize=20)
axis.set_ylabel('number of followers',fontsize=20)

#line_ort = axis.plot(year, ort, label = "real orthodox")
line_who = axis.plot(year, whole_diaspora, label = "diaspora jews", linewidth=10, linestyle="-", c="#994d00",
         solid_capstyle="round")
line_oth = axis.plot(year, oth, label = "global other", linewidth=10, linestyle="-", c="orange",
         solid_capstyle="round")
line_ort = axis.plot(year,ort, label = "global orthodox", linewidth=10, linestyle="-", c="black",
         solid_capstyle="round")
line_isr = axis.plot(year[1:],israeli_all, label = "israeli jews", linewidth=10, linestyle="-", c="#0052cc",
         solid_capstyle="round")


plt.legend(bbox_to_anchor=(1.05, 1), loc=2,
           ncol=1,prop={'size': 18}, borderaxespad=0.)
plt.show()
to_correlate=pd.DataFrame(np.hstack((whole_diaspora[1:],oth[1:],israeli_all,ort[1:])),columns=['Diaspora','"other"','Israeli','"orthodox"'])
f, ax = plt.subplots(figsize=(10, 8))
ax.set_title('Correlation ',fontsize=25)


corr = to_correlate.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)

for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(16) 
                # specify integer or one of preset strings, e.g.
                #tick.label.set_fontsize('x-small') 
              #  tick.label.set_rotation('vertical')
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(16) 
    tick.label.set_rotation('horizontal')
