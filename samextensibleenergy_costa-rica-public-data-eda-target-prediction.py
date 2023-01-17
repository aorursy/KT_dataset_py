import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import train_test_split
from plotly.offline import init_notebook_mode, iplot
from collections import Counter
import plotly.graph_objs as go
import plotly.plotly as py
from plotly import tools
from datetime import date
import plotly.figure_factory as ff

import random 
import warnings
import operator
warnings.filterwarnings("ignore")
init_notebook_mode(connected=True)
test = pd.read_csv('../input/test.csv')
train = pd.read_csv('../input/train.csv')

print ("Train Dataset: Rows, Columns: ", train.shape)
print ("Test Dataset: Rows, Columns: ", test.shape)
train.tail(n=3)
print ("Top Columns having missing values")
missmap = train.isnull().sum().to_frame().sort_values(0, ascending = False)
missmap.head()
d1 = train # keeping the original dataframe untouched
d1 = d1.fillna(value = 0) # Converting all the NaN values to 0
# checking the null value removal
missmap = d1.isnull().sum().to_frame().sort_values(0, ascending = False)
missmap.head()
def compare_plot(col, title):
#     tr1 = d1[d1['Target'] == 1][col].value_counts().to_dict()
#     tr2 = d1[d1['Target'] == 2][col].value_counts().to_dict()
#     tr3 = d1[d1['Target'] == 3][col].value_counts().to_dict()
#     tr4 = d1[d1['Target'] == 4][col].value_counts().to_dict()

        tar1 = dict(Counter(d1['Target']==1)) # generating a dictionary for counts of different target values
        tar2 = dict(Counter(d1['Target']==2))
        tar3 = dict(Counter(d1['Target']==3))
        tar4 = dict(Counter(d1['Target']==4))
      
        x_coord = ['Extereme', 'Moderate', 'Vulnerable', 'NonVulnerable'] # defining the x-co-ordinate variables
        trace1 = go.Bar(y=[tar1[0], tar2[0], tar3[0], tar4[0]], name="Not Present", x=x_coord, marker=dict(color="pink", opacity=0.6))
        trace2 = go.Bar(y=[tar1[1], tar2[1], tar3[1], tar4[1]], name="Present", x=x_coord, marker=dict(color="darkmagenta", opacity=0.6))
    
        return trace1, trace2 
    
tr1, tr2 = compare_plot("v18q", "Tablet")
tr3, tr4 = compare_plot("refrig", "Refrigirator")
tr5, tr6 = compare_plot("computer", "Computer")
tr7, tr8 = compare_plot("television", "Television")
tr9, tr10 = compare_plot("mobilephone", "MobilePhone")
titles = ["Tablet", "Refrigirator", "Computer", "Television", "MobilePhone"]

fig = tools.make_subplots(rows=3, cols=2, print_grid=False, subplot_titles=titles)
fig.append_trace(tr1, 1, 1)
fig.append_trace(tr2, 1, 1)
fig.append_trace(tr3, 1, 2)
fig.append_trace(tr4, 1, 2)
fig.append_trace(tr5, 2, 1)
fig.append_trace(tr6, 2, 1)
fig.append_trace(tr7, 2, 2)
fig.append_trace(tr8, 2, 2)
fig.append_trace(tr9, 3, 1)
fig.append_trace(tr10, 3, 1)

fig['layout'].update(height=1000, title="Electronic Gadgets v/s Household types", barmode="stack", showlegend=False)
iplot(fig)
def find_prominent(row, mats):
    for c in mats:
        if row[c] == 1:
            return c
    return 

def combine(starter, colname, title, replacemap):
    mats = [c for c in train.columns if c.startswith(starter)]
    train[colname] = train.apply(lambda row : find_prominent(row, mats), axis=1)
    train[colname] = train[colname].apply(lambda x : replacemap[x] if x != None else x )

    om1 = train[train['Target'] == 1][colname].value_counts().to_frame()
    om2 = train[train['Target'] == 2][colname].value_counts().to_frame()
    om3 = train[train['Target'] == 3][colname].value_counts().to_frame()
    om4 = train[train['Target'] == 4][colname].value_counts().to_frame()

    trace1 = go.Bar(y=om1[colname], x=om1.index, name="Extereme", marker=dict(color='red', opacity=0.9))
    trace2 = go.Bar(y=om2[colname], x=om2.index, name="Moderate", marker=dict(color='red', opacity=0.5))
    trace3 = go.Bar(y=om3[colname], x=om3.index, name="Vulnerable", marker=dict(color='green', opacity=0.5))
    trace4 = go.Bar(y=om4[colname], x=om4.index, name="NonVulnerable", marker=dict(color='green', opacity=0.9))
    return [trace1, trace2, trace3, trace4]

titles = ["Outside Wall Material", "Floor Material", "Roof Material", "Sanitary Conditions", "Cooking Energy Sources", "Disposal Methods"]
fig = tools.make_subplots(rows=3, cols=2, print_grid=False, subplot_titles=titles)


### outside material
flr = {'paredblolad' : "Block / Brick", "paredpreb" : "Cement", "paredmad" : "Wood",
      "paredzocalo" : "Socket", "pareddes" : "Waste Material", "paredfibras" : "Fibres",
      "paredother" : "Other", "paredzinc": "Zink"}
res = combine("pared", "outside_material", "Predominanat Material of the External Walls", flr)      
for x in res:
    fig.append_trace(x, 1, 1)

### floor material 
flr = {'pisomoscer' : "Mosaic / Ceramic", "pisocemento" : "Cement", "pisonatur" : "Natural Material",
      "pisonotiene" : "No Floor", "pisomadera" : "Wood", "pisoother" : "Other"}
res = combine("piso", "floor_material", "Floor Material of the Households", flr)
for x in res:
    fig.append_trace(x, 1, 2)

    
### Roof Material
flr = {'techozinc' : "Zinc", "techoentrepiso" : "Fibre / Cement", "techocane" : "Natural Fibre", "techootro" : "Other"}
res = combine("tech", "roof_material", "Roof Material of the Households", flr)  
for x in res:
    fig.append_trace(x, 2, 1)


### Sanitary Conditions
flr = {'sanitario1' : "No Toilet", "sanitario2" : "Sewer / Cesspool", "sanitario3" : "Septic Tank",
       "sanitario5" : "Black Hole", "sanitario6" : "Other System"}
res = combine("sanit", "sanitary", "Sanitary Conditions of the Households", flr)
for x in res:
    fig.append_trace(x, 2, 2)

### Energy Source
flr = {'energcocinar1' : "No Kitchen", "energcocinar2" : "Electricity", "energcocinar3" : "Cooking Gas",
       "energcocinar4" : "Wood Charcoal"}
res = combine("energ", "energy_source", "Main source of energy for cooking", flr)  
for x in res:
    fig.append_trace(x, 3, 1)

### Disposal Methods
flr = {"elimbasu1":"Tanker truck",
"elimbasu2": "Buried",
"elimbasu3": "Burning",
"elimbasu4": "Unoccupied space",
"elimbasu5": "River",
"elimbasu6": "Other"}
res = combine("elim", "waste_method", "Rubbish Disposals Method", flr)  
for x in res:
    fig.append_trace(x, 3, 2)

fig['layout'].update(height=900, title="Key Characteristics of Households", barmode="stack", showlegend=False)
iplot(fig)
def combine2(starter, colname, title, replacemap, plotme = True):
    mats = [c for c in train.columns if c.startswith(starter)]
    train[colname] = train.apply(lambda row : find_prominent(row, mats), axis=1)
    train[colname] = train[colname].apply(lambda x : replacemap[x] if x != None else x )

    om1 = train[train['Target'] == 1][colname].value_counts().to_frame()
    om2 = train[train['Target'] == 2][colname].value_counts().to_frame()
    om3 = train[train['Target'] == 3][colname].value_counts().to_frame()
    om4 = train[train['Target'] == 4][colname].value_counts().to_frame()

    trace1 = go.Bar(y=om1[colname], x=om1.index, name="Extereme", marker=dict(color='red', opacity=0.9))
    trace2 = go.Bar(y=om2[colname], x=om2.index, name="Moderate", marker=dict(color='orange', opacity=0.5))
    trace3 = go.Bar(y=om3[colname], x=om3.index, name="Vulnerable", marker=dict(color='yellow', opacity=0.9))
    trace4 = go.Bar(y=om4[colname], x=om4.index, name="NonVulnerable", marker=dict(color='green', opacity=0.5))

    data = [trace1, trace2, trace3, trace4]
    layout = dict(title=title, legend=dict(y=1.1, orientation="h"), barmode="stack", margin=dict(l=50), height=400)
    fig = go.Figure(data=data, layout=layout)
    if plotme:
        iplot(fig)


flr = {"instlevel1": "No Education", "instlevel2": "Incomplete Primary", "instlevel3": "Complete Primary", 
       "instlevel4": "Incomplete Sc.", "instlevel5": "Complete Sc.", "instlevel6": "Incomplete Tech Sc.",
       "instlevel7": "Complete Tech Sc.", "instlevel8": "Undergraduation", "instlevel9": "Postgraduation"}
combine2("instl", "education_details", "Education Details of Family Members", flr)  

flr = {"estadocivil1": "< 10 years", "estadocivil2": "Free / Coupled union", "estadocivil3": "Married", 
       "estadocivil4": "Divorced", "estadocivil5": "Separated", "estadocivil6": "Widow",
       "estadocivil7": "Single"}
combine2("estado", "status_members", "Status of Family Members", flr)  

flr = {"lugar1": "Central", "lugar2": "Chorotega", "lugar3": "PacÃƒÂ­fico central", 
       "lugar4": "Brunca", "lugar5": "Huetar AtlÃƒÂ¡ntica", "lugar6": "Huetar Norte"}
combine2("lugar", "region", "Region of the Households", flr)  





d1['Toilet_Score'] = (d1['sanitario1'] + 0 * (d1['sanitario2'] + d1['sanitario3'] + d1['sanitario5'] +  d1['sanitario6']))
# sanitario1, =1 no toilet in the dwelling
# sanitario2, =1 toilet connected to sewer or cesspool
# sanitario3, =1 toilet connected to  septic tank
# sanitario5, =1 toilet connected to black hole or letrine
# sanitario6, =1 toilet connected to other system

# only houses with no toilets are provided 1 point, others 'zero'
d1['Energy_Score'] = ( d1['energcocinar1'] +  (d1['energcocinar4'] ) - 1 * ( d1['energcocinar2'] + d1['energcocinar3'] ))
# energcocinar1, =1 no main source of energy used for cooking (no kitchen)
# energcocinar2, =1 main source of energy used for cooking electricity
# energcocinar3, =1 main source of energy used for cooking gas
# energcocinar4, =1 main source of energy used for cooking wood charcoal

# The No kitchen scenario and the wood charcoal scenario is given 1 point, 
# the other kitchen scenarios are given negative points to balance out the overall economic conditions
d1['Water_Score'] = ( (0 * d1['abastaguadentro']) + (0.5 * d1['abastaguafuera']) + (1 * d1['abastaguano']))
# abastaguadentro, =1 if water provision inside the dwelling
# abastaguafuera, =1 if water provision outside the dwelling
# abastaguano, =1 if no water provision

# for No water provision 1 point is allotted, for water provisionj outside the house 0.5 point system is awarded
d1['Electricity_Score'] = ( (0 * (d1['public'] + d1['planpri'] + d1['coopele']) ) + (1 * d1['noelec']) )
# public, =1 electricity from CNFL,  ICE,  ESPH/JASEC
# planpri, =1 electricity from private plant
# noelec, =1 no electricity in the dwelling
# coopele, =1 electricity from cooperative

#Only no electricity houses are awarded a single point
d1['Wall_Score'] = ((1*(d1['pareddes'] + d1['paredfibras'] + d1['paredzocalo'] )) + (0 * d1['paredblolad']) + (-1 * d1['paredpreb']))
# paredblolad, =1 if predominant material on the outside wall is block or brick
# paredzocalo, =1 if predominant material on the outside wall is socket (wood,  zinc or absbestos) --
# paredpreb, =1 if predominant material on the outside wall is prefabricated or cement
# pareddes, =1 if predominant material on the outside wall is waste material --
# paredfibras, =1 if predominant material on the outside wall is natural fibers --


d1['Floor_Score'] = ((1*(d1['pisonotiene'] + d1['pisonatur'])) + (0 * (d1['pisomadera'] + d1['pisoother'])) + (-1 * (d1['pisomoscer']+ d1['pisocemento'] )))

# pisomoscer, "=1 if predominant material on the floor is mosaic,  ceramic,  terrazo"
# pisocemento, =1 if predominant material on the floor is cement
# pisoother, =1 if predominant material on the floor is other
# pisonatur, =1 if predominant material on the floor is  natural material --
# pisonotiene, =1 if no floor at the household --
# pisomadera, =1 if predominant material on the floor is wood --

# If floor material is a natural material and/or no floor then 1 point is awarded
# If its of other or wood then 0 points are awarded
# If its cement or mosaic likes then a negative 1 point is awarded
d1['Roof_Score'] = ((1*(d1['techozinc'] + d1['techocane'])) + (0 * ( d1['techootro'])) + (-1 * (d1['techoentrepiso']+ d1['pisocemento'] )))

# techozinc, =1 if predominant material on the roof is metal foil or zink
# techoentrepiso, "=1 if predominant material on the roof is fiber cement,  mezzanine "
# techocane, =1 if predominant material on the roof is natural fibers
# techootro, =1 if predominant material on the roof is other
# cielorazo, =1 if the house has ceiling

# If roof material is a natural material and/or zinc then 1 point is awarded
# If its of other or wood then 0 points are awarded
# If its cement or mezzonine likes then a negative 1 point is awarded
# adding up all the amenities, Electtronics, House Condition Score
d1['Amenities_Score'] = d1['Electricity_Score'] + d1['Water_Score'] + d1['Energy_Score'] + d1['Toilet_Score']
d1['House_Condition_Score'] = d1['Roof_Score'] + d1['Floor_Score'] + d1['Wall_Score']
d1['Electronics_Score'] = (-1 * (d1['refrig'] + d1['computer'] + d1['television'] ) + (- 0.25 * d1['qmobilephone']))

# Geographic Diversification
ctrl = d1['lugar1'].sum()
chor = d1['lugar2'].sum()
pctl = d1['lugar3'].sum()
brun = d1['lugar4'].sum()
atlc = d1['lugar5'].sum()
nort = d1['lugar6'].sum()

# Data to plot
labels = 'Central', 'Chorotega', 'Pacafic Central', 'Brunca' , 'Atlantic'  , 'North'
sizes = [ctrl, chor, pctl, brun, atlc, nort]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'orange', 'darkturquoise']
explode = (0.05, 0, 0, 0,0.05,0)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode,  labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.show()
#Regional Diversification

urb = d1['area1'].sum()
rur = d1['area2'].sum()
print (urb,rur)

import pandas as pd
from matplotlib.pyplot import *

fig, ax = subplots()
df = pd.DataFrame({'urb': urb, 'rur' : rur}, index=['Counts'])
df.plot(kind='bar', ax=ax , color = ['darkmagenta', 'orchid'], edgecolor = 'black')
ax.legend(["Urban", "Rural"]);

# Converting rent per room and rent per capita to normalize
d1['Rent per room'] = d1['v2a1']/ d1['rooms']
d1['Per Capita Rent'] = d1['v2a1']/ (d1['overcrowding'] * d1['rooms'])

own_paid = d1['tipovivi1'].sum()
own_inst = d1['tipovivi2'].sum()
rent = d1['tipovivi3'].sum()
prec = d1['tipovivi4'].sum()
othr = d1['tipovivi5'].sum()


print (own_paid, own_inst, rent, prec, othr)

fig, ax = subplots()
df = pd.DataFrame({'own_paid': own_paid, 'own_inst' :own_inst,'rent': rent, 'prec' : prec,'othr': othr }, index=['Counts'])
df.plot(kind='bar', ax=ax , color = ['darkmagenta', 'orchid','deeppink','hotpink','crimson'], edgecolor = 'black')
ax.legend(["Own_Paid", "Own_installment", "Rented", "Precarious", "Other"]);

# It can be seen that most of the houses are own paid. and few of them are in installments and rented.
# dropping other factors where the house is not rented
d1_a = d1.dropna(subset=['Rent per room'])
d1_a.count()                       
x = d1_a['Rent per room']
plt.hist(x, bins=25, color = 'yellow', edgecolor = 'black')
plt.ylabel('Counts')
plt.xlabel('Price in Costa Rican Colon')
plt.show()
d1_a['Rent per room'] = d1_a['Rent per room'].round(0).astype(int)
d1_a_0 = d1_a[  (d1_a['Rent per room']<10001) ]
d1_a_1 = d1_a[ (d1_a['Rent per room']>10000) & (d1_a['Rent per room']<20001) ]
d1_a_2 = d1_a[ (d1_a['Rent per room']>20000) & (d1_a['Rent per room']<30001) ]
d1_a_3 = d1_a[ (d1_a['Rent per room']>30000) & (d1_a['Rent per room']<40001) ]
d1_a_4 = d1_a[ (d1_a['Rent per room']>40000) & (d1_a['Rent per room']<50001) ]
d1_a_5 = d1_a[ (d1_a['Rent per room']>50000) & (d1_a['Rent per room']<60001) ]
d1_a_6 = d1_a[ (d1_a['Rent per room']>60000) & (d1_a['Rent per room']<100001) ]
d1_a_7 = d1_a[  d1_a['Rent per room']>100001]

urb_0 = d1_a_0['area1'].sum()
rur_0 = d1_a_0['area2'].sum()
urb_1 = d1_a_1['area1'].sum()
rur_1 = d1_a_1['area2'].sum()
urb_2 = d1_a_2['area1'].sum()
rur_2 = d1_a_2['area2'].sum()
urb_3 = d1_a_3['area1'].sum()
rur_3 = d1_a_3['area2'].sum()
urb_4 = d1_a_4['area1'].sum()
rur_4 = d1_a_4['area2'].sum()
urb_5 = d1_a_5['area1'].sum()
rur_5 = d1_a_5['area2'].sum()
urb_6 = d1_a_6['area1'].sum()
rur_6 = d1_a_6['area2'].sum()
urb_7 = d1_a_7['area1'].sum()
rur_7 = d1_a_7['area2'].sum()

Urban = np.array([urb_0, urb_1, urb_2, urb_3,urb_4, urb_5,urb_6, urb_7 ])
Rural = np.array([rur_0, rur_1, rur_2, rur_3,rur_4, rur_5,rur_6, rur_7 ])
ind = np.arange(8)
width = 0.50 

p1 = plt.bar(ind, Urban, width, color='pink')
p2 = plt.bar(ind, Rural, width, color='#d62728', bottom=Urban)
plt.xlabel('Price Per room (in thousands)')
plt.ylabel('Counts')
plt.title('Counts v/s Price per room')
plt.xticks(ind, ('<10', '  10-20  ', '  20-30  ', '  30-40  ', '  40-50  ' , '  50-60  ', '  60-100  ', '  >100  '))
plt.legend((p1[0], p2[0]), ('Urban', 'Rural'))

plt.show()

d1_a.head()
df2 = d1[[ 'SQBmeaned','SQBdependency','SQBage', 'Amenities_Score', 'Electronics_Score', 'House_Condition_Score', 'area1', 'Target']]
df2 = df2.dropna()

df2.count()
sns.set(style="white")

# Generate a large random dataset
d = df2

# Compute the correlation matrix
corr = d.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

train.loc[: , "agesq"]
d1_deadly = d1[d1['tipovivi4']== 1]
d1_deadly.describe # we can see there are 434 houses which are listed precarious. Lets dive deeper into this

d1_deadly.head()
plt.figure(figsize=(10,7))
d1_deadly['epared1'].value_counts().plot.bar(alpha=0.10, color = 'Black', legend="a")
d1_deadly['eviv1'].value_counts().plot.bar(edgecolor = 'black',alpha=0.50, color = 'yellow', legend=True)
d1_deadly['etecho1'].value_counts().plot.bar(edgecolor = 'black',alpha=0.90, color = 'red', legend=True)
L=plt.legend()
L.get_texts()[0].set_text('Bad Walls')
L.get_texts()[1].set_text('Bad Floors')
L.get_texts()[2].set_text('Bad Roofs')
from sklearn.linear_model import Ridge


train_without_categoricals = train.select_dtypes(exclude=['object'])

from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
# train_without_categoricals = my_imputer.fit_transform(train_without_categoricals)


X = my_imputer.fit_transform( train_without_categoricals.iloc[:, :-1])
x_train, x_cv, y_train, y_cv = train_test_split(X,train_without_categoricals.iloc[:, -1])

## training the model

ridgeReg = Ridge(alpha=0.05, normalize=True)
ridgeReg.fit(x_train,y_train)
pred = ridgeReg.predict(x_cv)

#calculating mse
mse = np.mean((pred - y_cv)**2)  

## calculating score 

print ("Ridge_Score:", ridgeReg.score(x_cv,y_cv))
print("Ridge_MSE:", mse)
from sklearn.linear_model import Lasso

lassoReg = Lasso(alpha=0.3, normalize=True)

lassoReg.fit(x_train,y_train)

pred = lassoReg.predict(x_cv)

# calculating mse

mse = np.mean((pred - y_cv)**2)

print("Lasso_Score:", lassoReg.score(x_cv,y_cv))
print ("Lasso_mse:", mse)

from sklearn.linear_model import ElasticNet

ENreg = ElasticNet(alpha=1, l1_ratio=0.5, normalize=False)

ENreg.fit(x_train,y_train)

pred = ENreg.predict(x_cv)

#calculating mse

mse = np.mean((pred - y_cv)**2)

print("Eleastic_Net_mse:", mse)
print("Eleastic_Net_Score:", ENreg.score(x_cv,y_cv))

