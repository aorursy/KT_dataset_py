from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import re

from bokeh.io import output_notebook
from bokeh.sampledata import us_states
from bokeh.plotting import figure, show, output_file, ColumnDataSource
from bokeh.models import HoverTool, Range1d

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
opioids = pd.read_csv('../input/opioids.csv')
overdoses = pd.read_csv('../input/overdoses.csv')
prescribers = pd.read_csv('../input/prescriber-info.csv')
overdoses['Deaths'] = overdoses['Deaths'].apply(lambda x: float(re.sub(',', '', x)))
overdoses['Population'] = overdoses['Population'].apply(lambda x: float(re.sub(',', '', x)))
prescribers.head()
prescribers.describe()
ops = list(re.sub(r'[-\s]','.',x) for x in opioids.values[:,0])
prescribed_ops = list(set(ops) & set(prescribers.columns))

for i,drug in enumerate(prescribed_ops):
    print (i+1,drug)
# % of Opiod Prescribers
print (float(prescribers['Opioid.Prescriber'].sum())*100/prescribers.shape[0],"%")
prescribers['NumOpioids'] = prescribers.apply(lambda x: sum(x[prescribed_ops]),axis=1)
prescribers['NumPrescriptions'] = prescribers.apply(lambda x: sum(x.iloc[5:255]),axis=1)
prescribers['OpiodPrescribedVsPrescriptions'] = prescribers.apply(lambda x: float(x['NumOpioids'])/x['NumPrescriptions'],axis=1)
N = prescribers['NumOpioids'].shape[0]
x = prescribers['NumPrescriptions']
y = prescribers['NumOpioids']
colors = (192/255,192/255,192/255)
# area = np.pi*3
 
# Plot
# plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.scatter(x, y, c=colors, alpha=0.5)

m, b = np.polyfit(x, y, 1)
plt.plot(x, m*x + b, '-')

plt.title('Opiods Prescribed vs Number of Prescriptions')
plt.xlabel('Number of Opiods Prescribed')
plt.ylabel('Number of Prescriptions')
plt.show()
mu, sigma = np.mean(prescribers['OpiodPrescribedVsPrescriptions']), np.std(prescribers['OpiodPrescribedVsPrescriptions'])
n, bins, patches = plt.hist( prescribers['OpiodPrescribedVsPrescriptions'], 20, facecolor='grey', alpha=0.75)

plt.xlabel('Fraction of Opiods out of total drugs prescribed')
plt.ylabel('Number of Prescribers')
plt.title(r'$\mathrm{(Opiods Prescribed / Number of Preciptions)}\ \mu='+str(round(mu,2))+',\ \sigma='+str(round(sigma,2))+'$')
plt.grid(True)

plt.show()
OpioidPrescriber_OpioidFrac = prescribers.loc[prescribers['Opioid.Prescriber']>0,'OpiodPrescribedVsPrescriptions']

mu, sigma = np.mean(OpioidPrescriber_OpioidFrac), np.std(OpioidPrescriber_OpioidFrac)
n, bins, patches = plt.hist( OpioidPrescriber_OpioidFrac, 20, facecolor='grey', alpha=0.75)

plt.xlabel('Fraction of Opiods out of total drugs prescribed')
plt.ylabel('Number of Prescribers')
plt.title(r'$\mathrm{(Opiods Prescribed / Number of Preciptions)}\ \mu='+str(round(mu,2))+',\ \sigma='+str(round(sigma,2))+'$')
plt.grid(True)

plt.show()
fig = plt.figure(figsize=(7, 5))
axes = fig.add_subplot(1,1,1)
axes.boxplot( OpioidPrescriber_OpioidFrac, 0, 'rs', 0, 0.75, widths=[0.75])
plt.subplots_adjust(left=0.1, right=0.9, top=0.6, bottom=0.4)

plt.yticks([1],['Fraction of Opiods Prescribed by Opioid Prescribers'])
plt.show()
genderCount = np.array(list(prescribers[['Gender','NPI']].groupby('Gender').count()['NPI']))

genderCount = np.append([genderCount],[list(prescribers.loc[prescribers['Opioid.Prescriber']>0,['Gender','NPI']].groupby('Gender').count()['NPI'])], axis=0)

genderCount[0] = genderCount[0]-genderCount[1]

fig = plt.gcf()
fig.set_size_inches( 7, 5)

configs = genderCount[0]
N = configs.shape[0]
ind = np.arange(N)
width = 0.4

p1 = plt.bar(ind, genderCount[0], width, color='b')
p2 = plt.bar(ind, genderCount[1], width, bottom=genderCount[0], color='r')

# plt.ylim([0,120])
plt.yticks(fontsize=12)
plt.ylabel("Number of Prescribers", fontsize=12)
plt.xticks(ind,["Female","Male"])
plt.xlabel('Gender', fontsize=12)
plt.title("Opioid Prescribers by Gender")
plt.legend([p1[0], p2[0]], ["Did not prescribe opioids","Prescribed opioids"], fontsize=12, fancybox=True)
plt.show()
stateCount = pd.DataFrame(prescribers[['State','NPI']].groupby('State').count())

stateCount.reset_index(level=0, inplace=True)

stateCount.columns = ['State', 'Total_Prescribers']

stateCount_PrescribedOpiods = pd.DataFrame(prescribers.loc[prescribers['Opioid.Prescriber']>0,['State','NPI']].groupby('State').count())
stateCount_PrescribedOpiods.reset_index(level=0, inplace=True)
stateCount_PrescribedOpiods.columns = ['State', 'Opiod_Prescribers']
stateCount = pd.merge(stateCount, stateCount_PrescribedOpiods,  how='left', on="State")

stateCount = stateCount.fillna(0)

stateCount = stateCount.sort_values('Total_Prescribers')

fig = plt.gcf()
fig.set_size_inches( 20, 15)

N = stateCount.shape[0]
ind = np.arange(N)
width = 0.6

p1 = plt.bar(ind, stateCount['Total_Prescribers']-stateCount['Opiod_Prescribers'], width, color='b')
p2 = plt.bar(ind, stateCount['Opiod_Prescribers'], width, bottom=stateCount['Total_Prescribers']-stateCount['Opiod_Prescribers'], color='r')

# plt.ylim([0,120])
plt.yticks(fontsize=12)
plt.ylabel("Number of Prescribers", fontsize=15)
plt.xticks(ind,stateCount['State'], fontsize=15, rotation=70)
plt.xlabel('States', fontsize=15)
plt.title("Opioid Prescribers by State", fontsize=15)
plt.legend([p1[0], p2[0]], ["Did not prescribe opioids","Prescribed opioids"], fontsize=12, fancybox=True)
plt.show()
SpecialtyCount = pd.DataFrame(prescribers[['Specialty','NPI']].groupby('Specialty').count())

SpecialtyCount.reset_index(level=0, inplace=True)

SpecialtyCount.columns = ['Specialty', 'Total_Prescribers']

SpecialtyCount_PrescribedOpiods = pd.DataFrame(prescribers.loc[prescribers['Opioid.Prescriber']>0,['Specialty','NPI']].groupby('Specialty').count())
SpecialtyCount_PrescribedOpiods.reset_index(level=0, inplace=True)

SpecialtyCount_PrescribedOpiods.columns = ['Specialty', 'Opiod_Prescribers']
SpecialtyCount = pd.merge(SpecialtyCount, SpecialtyCount_PrescribedOpiods,  how='left', on="Specialty")

SpecialtyCount = SpecialtyCount.fillna(0)

SpecialtyCount = SpecialtyCount.sort_values('Total_Prescribers')

SpecialtyCount = SpecialtyCount[-30::]

fig = plt.gcf()
fig.set_size_inches( 20, 10)

N = SpecialtyCount.shape[0]
ind = np.arange(N)
width = 0.6

p1 = plt.bar(ind, SpecialtyCount['Total_Prescribers']-SpecialtyCount['Opiod_Prescribers'], width, color='b')
p2 = plt.bar(ind, SpecialtyCount['Opiod_Prescribers'], width, bottom=SpecialtyCount['Total_Prescribers']-SpecialtyCount['Opiod_Prescribers'], color='r')

# plt.ylim([0,120])
plt.yticks(fontsize=12)
plt.ylabel("Number of Prescribers", fontsize=15)
plt.xticks(ind,SpecialtyCount['Specialty'], fontsize=15, rotation=90)
plt.xlabel('Specialty', fontsize=15)
plt.title("Opioid Prescribers by Specialty (Top 30)", fontsize=15)
plt.legend([p1[0], p2[0]], ["Did not prescribe opioids","Prescribed opioids"], fontsize=12, fancybox=True)
plt.show()
SpecialtyCount = pd.DataFrame(prescribers[['Specialty','NPI']].groupby('Specialty').count())

SpecialtyCount.reset_index(level=0, inplace=True)

SpecialtyCount.columns = ['Specialty', 'Total_Prescribers']

SpecialtyCount_PrescribedOpiods = pd.DataFrame(prescribers.loc[prescribers['Opioid.Prescriber']>0,['Specialty','NPI']].groupby('Specialty').count())
SpecialtyCount_PrescribedOpiods.reset_index(level=0, inplace=True)

SpecialtyCount_PrescribedOpiods.columns = ['Specialty', 'Opiod_Prescribers']
SpecialtyCount = pd.merge(SpecialtyCount, SpecialtyCount_PrescribedOpiods,  how='left', on="Specialty")

SpecialtyCount = SpecialtyCount.fillna(0)

SpecialtyCount = SpecialtyCount.sort_values('Opiod_Prescribers')

SpecialtyCount = SpecialtyCount[-30::]

fig = plt.gcf()
fig.set_size_inches( 20, 10)

N = SpecialtyCount.shape[0]
ind = np.arange(N)
width = 0.6

p1 = plt.bar(ind, SpecialtyCount['Total_Prescribers']-SpecialtyCount['Opiod_Prescribers'], width, color='b')
p2 = plt.bar(ind, SpecialtyCount['Opiod_Prescribers'], width, bottom=SpecialtyCount['Total_Prescribers']-SpecialtyCount['Opiod_Prescribers'], color='r')

# plt.ylim([0,120])
plt.yticks(fontsize=12)
plt.ylabel("Number of Prescribers", fontsize=15)
plt.xticks(ind,SpecialtyCount['Specialty'], fontsize=15, rotation=90)
plt.xlabel('Specialty', fontsize=15)
plt.title("Opioid Prescribers by Specialty (Top 30)", fontsize=15)
plt.legend([p1[0], p2[0]], ["Did not prescribe opioids","Prescribed opioids"], fontsize=12, fancybox=True)
plt.show()
specialty = pd.DataFrame(prescribers.groupby(['Specialty']).count()['NPI']).sort_values('NPI')

specialty.loc[specialty['NPI']<40].shape


rareSpecialty = list(specialty.loc[specialty['NPI']<40].index)


prescribers.loc[prescribers['Specialty'].isin(rareSpecialty),'Specialty'] = prescribers.loc[prescribers['Specialty'].isin(rareSpecialty),'Specialty'].apply(lambda x: 'Surgery' if 'Surgery' in list(x.split( )) else 'Other')

prescribersData = prescribers.drop( ['NPI','Credentials'], axis=1)

prescribersData = pd.get_dummies(prescribersData, columns=['Gender','Specialty','State'], drop_first=True)
#convert it to numpy arrays
X= prescribersData.values

scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

pca = PCA(n_components=300)

pca.fit(X_scaled)

#The amount of variance that each PC explains
var= pca.explained_variance_ratio_

#Cumulative Variance explains
cum_var = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
plt.plot(var, color='y')
# plt.plot(cum_var, color='r')
# plt.xticks()
plt.ylabel('Principal Components')
plt.ylabel('Proportion of Variance Explained')
plt.title('Scree Plot')
#Looking at above plot I'm taking 20 variables
pca = PCA(n_components=20)

# pca.fit(X_scaled)
X1=pca.fit_transform(X_scaled)
print ("Explained variance by component: %s" % pca.explained_variance_ratio_)
print ("Variance explained by first 10 factors: %s" % (pca.explained_variance_ratio_[0:9].sum()/pca.explained_variance_ratio_.sum()))
print ("Since these explain ~80% of the variance they are selected for further analysis")
newFactors = pd.DataFrame(pca.components_,columns=prescribersData.columns)
newFactors = newFactors.loc[0:9]
newFactors
impFactors = list(set(pd.DataFrame(newFactors.max())[pd.DataFrame(newFactors.max()> 0.2)[0]].index).union(set(pd.DataFrame(newFactors.min())[pd.DataFrame(newFactors.min()< -0.2)[0]].index)))
pd.DataFrame(impFactors)
newFactors_ = newFactors[impFactors]
import seaborn as sns
%matplotlib inline
sns.set(rc={'figure.figsize':(12,25)})
# sns.heatmap(newFactors.T, cmap='RdYlGn', linewidths=0.5, annot=True)
sns.heatmap(newFactors_.T, cmap='RdYlGn', linewidths=0.5)
plt.plot()