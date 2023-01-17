import pandas as pd

import numpy as np
pd.set_option('display.max_rows', 100)

pd.set_option('display.max_columns', 500)
train=pd.read_csv('../input/costa-rican-household-poverty-prediction/train.csv')

train
train['Target'].value_counts()
train.shape
work=train
#code for number of null values in a column

for i in list(work.columns):

    count=0

    for j in  list(work[i].isnull()):

        if j is True:

            count=count+1

    if count != 0:

        print(i,'-',count)
work.loc[((work['v2a1'].isnull()) & (work['Target']==1) ),'v2a1']=work['v2a1'].where(work['Target']==1).dropna().median()

work.loc[((work['v2a1'].isnull()) & (work['Target']==2) ),'v2a1']=work['v2a1'].where(work['Target']==2).dropna().median()

work.loc[((work['v2a1'].isnull()) & (work['Target']==3) ),'v2a1']=work['v2a1'].where(work['Target']==3).dropna().median()

work.loc[((work['v2a1'].isnull()) & (work['Target']==4) ),'v2a1']=work['v2a1'].where(work['Target']==4).dropna().median()
work.loc[((work['v18q1'].isnull()) & (work['v18q']==0)),'v18q1']=0
ID=work['Id']

monthly_rent_payment=work['v2a1']

bathrooms=work['rooms']-work['bedrooms']

bedrooms=work['bedrooms']

refrigerator=work['refrig']

no_of_tablets=work['v18q1']

males_younger_12=work['r4h1']

males_older_12=work['r4h2']

female_younger_12=work['r4m1']

female_older_12=work['r4m2']

household_size=work['hhsize']

years_of_schooling=work['escolari']

material_outside_wall=work['paredblolad']+2*work['paredzocalo']+3*work['paredpreb']+4*work['pareddes']+5*work['paredmad']+6*work['paredzinc']+7*work['paredfibras']+8*work['paredother']

floor_material=work['pisomoscer']+2*work['pisocemento']+3*work['pisoother']+4*work['pisonatur']+5*work['pisonotiene']+6*work['pisomadera']

roof_material=work['techozinc']+2*work['techoentrepiso']+3*work['techocane']+4*work['techootro']

water_provision=work['abastaguano']+2*work['abastaguadentro']+3*work['abastaguafuera']

electricity=work['public']+2*work['planpri']+3*work['noelec']+4*work['coopele']

toilet=work['sanitario1']+2*work['sanitario2']+3*work['sanitario3']+4*work['sanitario5']+5*work['sanitario6']

cooking_energy=work['energcocinar1']+2*work['energcocinar2']+3*work['energcocinar3']+4*work['energcocinar4']

rubbish_disposal=work['elimbasu1']+2*work['elimbasu2']+3*work['elimbasu3']+4*work['elimbasu4']+5*work['elimbasu5']+6*work['elimbasu6']

condition_walls=work['epared1']+2*work['epared2']+3*work['epared3']

condition_floor=work['etecho1']+2*work['etecho2']+3*work['etecho3']

condition_roof=work['eviv1']+2*work['eviv2']+3*work['eviv3']
disable_person=work['dis']

gender=work['male']+2*work['female']

marital_status=work['estadocivil1']+2*work['estadocivil2']+3*work['estadocivil3']+4*work['estadocivil4']+5*work['estadocivil5']+6*work['estadocivil6']+7*work['estadocivil7']

in_house_position=work['parentesco1']+2*work['parentesco2']+3*work['parentesco3']+4*work['parentesco4']+5*work['parentesco5']+6*work['parentesco6']+7*work['parentesco7']+8*work['parentesco8']+9*work['parentesco9']+10*work['parentesco10']+11*work['parentesco11']+12*work['parentesco12']

Household_level_identifier=work['idhogar']

children_0_to_19=work['hogar_nin']

adult_below_65=work['hogar_adul']-work['hogar_mayor']

adult_above_65=work['hogar_mayor']
work.loc[(work['dependency']=='yes'), 'dependency']=1

work.loc[(work['dependency']=='no'), 'dependency']=0

dependency=work['dependency']

work.loc[(work['edjefe']=='no'),'edjefe']=0

work.loc[(work['edjefe']=='yes'),'edjefe']=1

work.loc[(work['edjefa']=='no'),'edjefa']=0

work.loc[(work['edjefa']=='yes'),'edjefa']=1



edu_male_head_years=work['edjefe']

edu_female_head_years=work['edjefa']

edu_avg_above_18=work['meaneduc']

education_level=work['instlevel1']+2*work['instlevel2']+3*work['instlevel3']+4*work['instlevel4']+5*work['instlevel6']+6*work['instlevel7']+7*work['instlevel7']+8*work['instlevel8']+9*work['instlevel9']
overcrowding=work['overcrowding']
house_owned_status=work['tipovivi1']+2*work['tipovivi2']+3*work['tipovivi3']+4*work['tipovivi4']+5*work['tipovivi5']
computer=work['computer']
television=work['television']
mobilephone=work['mobilephone']
no_of_mobile_phone=work['qmobilephone']
region=work['lugar1']+2*work['lugar2']+3*work['lugar3']+4*work['lugar4']+5*work['lugar5']+6*work['lugar6']
area=work['area1']+2*work['area2']

age=work['age']
about_person=['ID','age','years_of_schooling','disable_person','gender','marital_status','in_house_position','education_level']

about_household=['region','area','house_owned_status','monthly_rent_payment','bedrooms','bathrooms','condition_roof','condition_floor','condition_walls','material_outside_wall','floor_material','roof_material','water_provision','electricity','toilet','cooking_energy','rubbish_disposal']

gadget_in_household=['refrigerator','no_of_tablets','computer','television','mobilephone','no_of_mobile_phone']

household_composition=['Household_level_identifier','household_size','males_younger_12','males_older_12','female_younger_12','female_older_12','children_0_to_19','adult_below_65','adult_above_65']

education_level_in_house=['edu_male_head_years','edu_female_head_years','edu_avg_above_18']

household_stats=['overcrowding','dependency']
final=pd.DataFrame({ 'ID':ID, 'age':age, 'years_of_schooling':years_of_schooling, 'disable_person':disable_person, 'gender':gender, 'marital_status':marital_status, 'in_house_position':in_house_position, 'education_level':education_level, 'region':region, 'area':area, 'house_owned_status':house_owned_status, 'monthly_rent_payment':monthly_rent_payment, 'bedrooms':bedrooms, 'bathrooms':bathrooms, 'condition_roof':condition_roof, 'condition_floor':condition_floor, 'condition_walls':condition_walls, 'material_outside_wall':material_outside_wall, 'floor_material':floor_material, 'roof_material':roof_material, 'water_provision':water_provision, 'electricity':electricity, 'toilet':toilet, 'cooking_energy':cooking_energy, 'rubbish_disposal':rubbish_disposal, 'Household_level_identifier':Household_level_identifier, 'household_size':household_size, 'males_younger_12':males_younger_12, 'males_older_12':males_older_12, 'female_younger_12':female_younger_12, 'female_older_12':female_older_12, 'children_0_to_19':children_0_to_19, 'adult_below_65':adult_below_65, 'adult_above_65':adult_above_65, 'edu_male_head_years':edu_male_head_years, 'edu_female_head_years':edu_female_head_years, 'edu_avg_above_18':edu_avg_above_18, 'refrigerator':refrigerator, 'no_of_tablets':no_of_tablets, 'computer':computer, 'television':television, 'mobilephone':mobilephone, 'no_of_mobile_phone':no_of_mobile_phone, 'overcrowding':overcrowding, 'dependency':dependency})
final['Target']=work['Target']
final
final.shape
#code for number of null values in a column

for i in list(final.columns):

    count=0

    for j in  list(final[i].isnull()):

        if j is True:

            count=count+1

    if count != 0:

        print(i,'-',count)
final.loc[(final['edu_avg_above_18'].isnull() ),['ID','age','gender','household_size','Household_level_identifier','education_level','edu_male_head_years','edu_female_head_years','edu_avg_above_18','Target']]
final.loc[(final['ID']=='ID_bd8e11b0f' )]=final.loc[((final['household_size']==1) & (final['Target']==4)),'edu_avg_above_18'].median()

final.loc[(final['ID']=='ID_46ff87316' )]=final.loc[((final['household_size']==2) & (final['Target']==4)),'edu_avg_above_18'].median()

final.loc[(final['ID']=='ID_69f50bf3e' )]=final.loc[((final['household_size']==2) & (final['Target']==4)),'edu_avg_above_18'].median()

final.loc[(final['ID']=='ID_db3168f9f' )]=final.loc[((final['household_size']==2) & (final['Target']==4)),'edu_avg_above_18'].median()

final.loc[(final['ID']=='ID_2a7615902' )]=final.loc[((final['household_size']==2) & (final['Target']==4)),'edu_avg_above_18'].median()

final.loc[[1291,1840,1841,2049,2050],'edu_avg_above_18']
#code for number of null values in a column

for i in list(final.columns):

    count=0

    for j in  list(final[i].isnull()):

        if j is True:

            count=count+1

    if count != 0:

        print(i,'-',count)
final.to_csv('final.csv',index=False)
final.info()
for i in range(0,len(final['Target'])):

    if final.loc[i,'Target']==10:

        final.loc[i,'Target']=4

final['Target'].value_counts()   

final.to_csv('final.csv',index=False)
import seaborn as sns

import matplotlib.pyplot as plt

from scipy import stats
sns.set(color_codes=True)
final['age'].describe()
sns.distplot(final['age'], bins=5, kde=False, rug=True);

sns.kdeplot(final['age'], shade=True);
sns.set(style="whitegrid")

ax = sns.boxplot(x=final['age'])
final['years_of_schooling'].describe()
sns.distplot(final['years_of_schooling'], bins=5, kde=False, rug=True);
sns.set(style="whitegrid")

ax = sns.boxplot(x=final['years_of_schooling'])
sns.distplot(final['disable_person'],bins=2, kde=False, rug=True,color='blue');
final['disable_person'].value_counts()
sns.distplot(final['gender'], kde=False, rug=True,color='red');

final['gender'].value_counts()
plt.hist(final['marital_status'])

plt.xlabel('marital_status')

legends=['1-less than 10 years old','2-free or coupled uunion','3-married','4-divorced','5-seperated','6-widower','7-single']

plt.show()

legends

plt.hist(final['in_house_position'])

plt.xlabel('in_house_position')

legends=['1-household head','2-spouse/partner','3-son/doughter','4-stepson/doughter','5-son/doughter in law','6-grandson/doughter','7-mother/father','8-father/mother in law','9-brother/sister','10-brother/sister in law','11-other family member','10-other non family member']

plt.show()

legends
plt.hist(final['education_level'])

plt.xlabel('education_level')

legends=['1-no level of education','2-incomplete primary','3-complete primary','4-incomplete academic secondary level','5-complete academic secondary level','6-incomplete technical secondary level','7-complete technical secondary level','8-undergraduate and higher education','9-postgraduate higher education']

plt.show()

legends
plt.hist(final['region'])

plt.xlabel('region')

legends=['1-region Central','2-region Chorotega','3-region PacÃƒÂ­fico central','4-region Brunca','5-region Huetar AtlÃƒÂ¡ntica','6-region Huetar Norte']

plt.show()

legends
plt.hist(final['area'])

plt.xlabel('area')

legends=['1-zona urbana','2-zona rural']

plt.show()

legends
plt.hist(final['house_owned_status'])

plt.xlabel('house_owned_status')

legends=['1-own and fully paid house','2-own,  paying in installments','3-rented','4-precarious','5-other(assigned,  borrowed)']

plt.show()

legends
final['monthly_rent_payment'].describe()
sns.kdeplot(final['monthly_rent_payment'], shade=True);
sns.set(style="whitegrid")

ax = sns.boxplot(x=final['monthly_rent_payment'])
final['monthly_rent_payment']=final['monthly_rent_payment']/(np.max(final['monthly_rent_payment'])-np.min(final['monthly_rent_payment']))
whisker_max_limit=np.percentile(final['monthly_rent_payment'],75)+1.5*(np.percentile(final['monthly_rent_payment'],75)-np.percentile(final['monthly_rent_payment'],25))

whisker_max_limit
final['monthly_rent_payment'].where(final['monthly_rent_payment']<=whisker_max_limit).describe()
sns.distplot( final['monthly_rent_payment'].where(final['monthly_rent_payment']<=whisker_max_limit).dropna(),bins=6,kde=False, rug=True);
sns.kdeplot(final['monthly_rent_payment'].where(final['monthly_rent_payment']<=whisker_max_limit).dropna(), shade=True);
sns.distplot(final['bathrooms'],bins=4,kde=False, rug=True);
sns.distplot(final['bedrooms'],bins=4,kde=False, rug=True);
plt.hist(final['condition_walls'])

plt.xlabel('condition_walls')

legends=['1-walls are bad','2-walls are regular','3-walls are good']

plt.show()

legends
plt.hist(final['condition_floor'])

plt.xlabel('condition_floor')

legends=['1-floor are bad','2-floor are regular','3-floor are good']

plt.show()

legends
plt.hist(final['condition_roof'])

plt.xlabel('condition_roof')

legends=['1-roof are bad','2-roof are regular','3-roof are good']

plt.show()

legends
plt.hist(final['electricity'])

plt.xlabel('electricity')

legends=['1-electricity from CNFL,  ICE,  ESPH/JASEC','2-electricity from private plant','3-no electricity in the dwelling','4-electricity from cooperative']

plt.show()

legends
final['electricity'].value_counts()
final['household_size'].describe()
plt.hist(final['household_size'])

plt.xlabel('no_of_persons_in_household')

plt.show()

sns.set(style="whitegrid")

ax = sns.boxplot(x=final['household_size'])
plt.hist(final['males_younger_12'])

plt.xlabel('males_younger_12')

plt.ylabel("no_of_families")

plt.show()

final['males_younger_12'].value_counts()
plt.hist(final['males_older_12'])

plt.xlabel('males_older_12')

plt.ylabel("no_of_families")

plt.show()

final['males_older_12'].value_counts()
plt.hist(final['female_younger_12'])

plt.xlabel('female_younger_12')

plt.ylabel("no_of_families")

plt.show()

final['female_younger_12'].value_counts()
plt.hist(final['female_older_12'])

plt.xlabel('female_older_12')

plt.ylabel("no_of_families")

plt.show()

final['female_older_12'].value_counts()
final['female_older_12'].value_counts()
plt.hist(final['children_0_to_19'])

plt.xlabel('children_0_to_19')

plt.ylabel("no_of_families")

plt.show()

final['children_0_to_19'].value_counts()
plt.hist(final['adult_below_65'])

plt.xlabel('adult_below_65')

plt.ylabel("no_of_families")

plt.show()

final['adult_below_65'].value_counts()
plt.hist(final['adult_above_65'])

plt.xlabel('adult_above_65')

plt.ylabel("no_of_families")

plt.show()

final['adult_above_65'].value_counts()
plt.hist([final['refrigerator'],final['computer'],final['television'],final['mobilephone']],bins=2)

plt.ylabel("no_of_families")

legends=['refrigerator','computer','television','mobilephone']

plt.legend(legends)

plt.show()

plt.hist(final['overcrowding'])

plt.ylabel("no_of_families")

plt.show()
corr_matrix=final.corr()
corr_matrix['adult_above_65'].sort_values(ascending=False)
corr = final.corr()

ax = sns.heatmap(

    corr, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=True

)



ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=45,

    horizontalalignment='right'

);

final['condition_walls'].corr(final['roof_material'])
from sklearn.metrics import confusion_matrix 

from sklearn.tree import DecisionTreeClassifier 

from sklearn.metrics import accuracy_score 

from sklearn.metrics import classification_report

from sklearn.metrics import roc_curve

from sklearn.tree import DecisionTreeRegressor

import seaborn as sns

import matplotlib.pyplot as plt

import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation

from keras.optimizers import SGD

import numpy as np

import pandas as pd

import tensorflow

from keras.utils import to_categorical

from sklearn.datasets import fetch_mldata

from sklearn.preprocessing import StandardScaler

from sklearn import metrics

from sklearn.linear_model import LogisticRegression
pd.set_option('display.max_columns', None)

final=pd.read_csv('../input/final-data/final.csv')

work=final.copy(deep=True)

work
def plot_confusion_matrix(cm, names, title='Confusion matrix', cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(names))

    plt.xticks(tick_marks, names, rotation=45)

    plt.yticks(tick_marks, names)

    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')



def cal_accuracy(y_test, y_pred): 

      

    print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred)) 

    

    products=[1,2,3,4]

    cm = confusion_matrix(y_test, result)

    np.set_printoptions(precision=2)

    plt.figure()

    plot_confusion_matrix(cm, products)

    plt.show()

      

    print ("Accuracy : ", 

    accuracy_score(y_test,y_pred)*100)

    

    print ("Sensitivity for 1 : ", (cm[0,0]/sum(cm[0]))*100)

    print ("Sensitivity for 2 : ", (cm[1,1]/sum(cm[1]))*100)

    print ("Sensitivity for 3 : ", (cm[2,2]/sum(cm[2]))*100)

    print ("Sensitivity for 4 : ", (cm[3,3]/sum(cm[3]))*100)
data = work[['age', 'years_of_schooling', 'disable_person', 'gender',

       'marital_status', 'in_house_position', 'education_level', 'region',

       'area', 'house_owned_status', 'monthly_rent_payment', 'bedrooms',

       'bathrooms', 'condition_roof', 'condition_floor', 'condition_walls',

       'material_outside_wall', 'floor_material', 'roof_material',

       'water_provision', 'electricity', 'toilet', 'cooking_energy',

       'rubbish_disposal', 'household_size',

       'males_younger_12', 'males_older_12', 'female_younger_12',

       'female_older_12', 'children_0_to_19', 'adult_below_65',

       'adult_above_65', 'edu_male_head_years', 'edu_female_head_years',

       'edu_avg_above_18', 'refrigerator', 'no_of_tablets', 'computer',

       'television', 'mobilephone', 'no_of_mobile_phone', 'overcrowding',

       'dependency']].copy(deep=True)



x_train=data.loc[1:8557,].values

x_test=data.loc[8557:9556,].values

y_train=work.loc[1:8557,'Target'].values.astype(int)

y_test=work.loc[8557:9556,'Target'].values.astype(int)

clf_gini = DecisionTreeClassifier( criterion = "gini", max_depth = None)

clf_gini.fit(x_train, y_train)

result=clf_gini.predict(x_test)

cal_accuracy(y_test, result)

clf_entropy = DecisionTreeClassifier( criterion = "entropy", max_depth = None)

clf_entropy.fit(x_train, y_train)

result=clf_entropy.predict(x_test)

cal_accuracy(y_test, result)
data = work[['age', 'disable_person', 'gender',

       'marital_status', 'education_level', 'region',

       'area', 'house_owned_status', 'monthly_rent_payment', 

       'condition_roof', 'condition_floor', 'condition_walls',

        'water_provision', 'electricity', 'toilet', 'cooking_energy',

       'rubbish_disposal', 'household_size',

        'refrigerator', 'no_of_tablets', 'computer',

       'television', 'no_of_mobile_phone', 'overcrowding',

       'dependency']].copy(deep=True)





x_train=data.loc[1:8557,].values

x_test=data.loc[8557:9556,].values

y_train=work.loc[1:8557,'Target'].values.astype(int)

y_test=work.loc[8557:9556,'Target'].values.astype(int)
clf_gini = DecisionTreeClassifier( criterion = "gini", max_depth = None)

clf_gini.fit(x_train, y_train)

result=clf_gini.predict(x_test)

cal_accuracy(y_test, result)
clf_entropy = DecisionTreeClassifier( criterion = "entropy", max_depth = None)

clf_entropy.fit(x_train, y_train)

result=clf_entropy.predict(x_test)

cal_accuracy(y_test, result)
data = work[['age', 'years_of_schooling', 'disable_person', 'gender',

       'marital_status', 'in_house_position', 'education_level', 'region',

       'area', 'house_owned_status', 'monthly_rent_payment', 'bedrooms',

       'bathrooms', 'condition_roof', 'condition_floor', 'condition_walls',

       'material_outside_wall', 'floor_material', 'roof_material',

       'water_provision', 'electricity', 'toilet', 'cooking_energy',

       'rubbish_disposal', 'household_size',

       'males_younger_12', 'males_older_12', 'female_younger_12',

       'female_older_12', 'children_0_to_19', 'adult_below_65',

       'adult_above_65', 'edu_male_head_years', 'edu_female_head_years',

       'edu_avg_above_18', 'refrigerator', 'no_of_tablets', 'computer',

       'television', 'mobilephone', 'no_of_mobile_phone', 'overcrowding',

       'dependency']].copy(deep=True)





x_train=data.loc[1:8557,].values

x_test=data.loc[8557:9556,].values

y_train=work.loc[1:8557,'Target'].values.astype(int)

y_test=work.loc[8557:9556,'Target'].values.astype(int)



scaler = StandardScaler()

scaler.fit(x_train)



x_train = scaler.transform(x_train)

x_test = scaler.transform(x_test)



model = LogisticRegression(solver = 'lbfgs',multi_class='multinomial')

model.fit(x_train, y_train)



result = model.predict(x_test)



cal_accuracy(y_test, result)

data = work[['age', 'disable_person', 'gender',

       'marital_status', 'education_level', 'region',

       'area', 'house_owned_status', 'monthly_rent_payment', 

       'condition_roof', 'condition_floor', 'condition_walls',

        'water_provision', 'electricity', 'toilet', 'cooking_energy',

       'rubbish_disposal', 'household_size',

        'refrigerator', 'no_of_tablets', 'computer',

       'television', 'no_of_mobile_phone', 'overcrowding',

       'dependency']].copy(deep=True)



x_train=data.loc[1:8557,].values

x_test=data.loc[8557:9556,].values

y_train=work.loc[1:8557,'Target'].values.astype(int)

y_test=work.loc[8557:9556,'Target'].values.astype(int)



scaler = StandardScaler()

scaler.fit(x_train)



x_train = scaler.transform(x_train)

x_test = scaler.transform(x_test)



model = LogisticRegression(solver = 'lbfgs',multi_class='multinomial')

model.fit(x_train, y_train)



result = model.predict(x_test)



cal_accuracy(y_test, result)
data = work[['age', 'years_of_schooling', 'disable_person', 'gender',

       'marital_status', 'in_house_position', 'education_level', 'region',

       'area', 'house_owned_status', 'monthly_rent_payment', 'bedrooms',

       'bathrooms', 'condition_roof', 'condition_floor', 'condition_walls',

       'material_outside_wall', 'floor_material', 'roof_material',

       'water_provision', 'electricity', 'toilet', 'cooking_energy',

       'rubbish_disposal', 'household_size',

       'males_younger_12', 'males_older_12', 'female_younger_12',

       'female_older_12', 'children_0_to_19', 'adult_below_65',

       'adult_above_65', 'edu_male_head_years', 'edu_female_head_years',

       'edu_avg_above_18', 'refrigerator', 'no_of_tablets', 'computer',

       'television', 'mobilephone', 'no_of_mobile_phone', 'overcrowding',

       'dependency']].copy(deep=True)





x_train=data.loc[1:8557,].values

x_test=data.loc[8557:9556,].values

y_train=work.loc[1:8557,'Target'].values.astype(int)

y_test=work.loc[8557:9556,'Target'].values.astype(int)



from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test= sc.fit_transform(x_test)



y1_train=np.zeros((y_train.shape[0],4), dtype=np.int)

for i in range(0,y_train.shape[0]):

    if y_train[i]==1:

        y1_train[i][0]=1

    if y_train[i]==2:

        y1_train[i][1]=1

    if y_train[i]==3:

        y1_train[i][2]=1

    if y_train[i]==4:

        y1_train[i][3]=1



y1_test=np.zeros((y_test.shape[0],4), dtype=np.int)

for i in range(0,y_test.shape[0]):

    if y_test[i]==1:

        y1_test[i][0]=1

    if y_test[i]==2:

        y1_test[i][1]=1

    if y_test[i]==3:

        y1_test[i][2]=1

    if y_test[i]==4:

        y1_test[i][3]=1



model=Sequential()



model.add(Dense(int((data.shape[1]+1)/2), kernel_initializer="uniform", activation = 'relu', input_dim = data.shape[1]))

model.add(Dense(int((data.shape[1]+1)/2), kernel_initializer="uniform", activation = 'relu'))

model.add(Dense(4, kernel_initializer="uniform", activation = 'softmax'))



sgd=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)



model.compile(loss='categorical_crossentropy',

             optimizer=sgd,

             metrics=['accuracy'])



model.fit(x_train,y1_train,epochs=1000, batch_size=128)
result = model.predict(x_test)

result=np.argmax(result, axis = 1)+1

cal_accuracy(y_test, result)
data = work[[ 'monthly_rent_payment',

       'area','household_size',

         'no_of_tablets', 

        'no_of_mobile_phone', 'overcrowding',

       'dependency']].copy(deep=True)





x_train=data.loc[1:8557,].values

x_test=data.loc[8557:9556,].values

y_train=work.loc[1:8557,'Target'].values.astype(int)

y_test=work.loc[8557:9556,'Target'].values.astype(int)





from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test= sc.fit_transform(x_test)



y1_train=np.zeros((y_train.shape[0],4), dtype=np.int)

for i in range(0,y_train.shape[0]):

    if y_train[i]==1:

        y1_train[i][0]=1

    if y_train[i]==2:

        y1_train[i][1]=1

    if y_train[i]==3:

        y1_train[i][2]=1

    if y_train[i]==4:

        y1_train[i][3]=1



y1_test=np.zeros((y_test.shape[0],4), dtype=np.int)

for i in range(0,y_test.shape[0]):

    if y_test[i]==1:

        y1_test[i][0]=1

    if y_test[i]==2:

        y1_test[i][1]=1

    if y_test[i]==3:

        y1_test[i][2]=1

    if y_test[i]==4:

        y1_test[i][3]=1



model=Sequential()



model.add(Dense(output_dim = int((data.shape[1]+1)/2), init = 'uniform', activation = 'relu', input_dim = data.shape[1]))

model.add(Dense(output_dim = int((data.shape[1]+1)/2), init = 'uniform', activation = 'relu'))

model.add(Dense(output_dim = 4, init = 'uniform', activation = 'softmax'))





sgd=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)



model.compile(loss='categorical_crossentropy',

             optimizer=sgd,

             metrics=['accuracy'])



model.fit(x_train,y1_train,epochs=1000, batch_size=128)
res = model.predict(x_test)

result=np.argmax(res, axis = 1)+1

cal_accuracy(y_test, result)
Decesion_gini=pd.DataFrame([

 [ 78,   2,   1,   1],

 [  9, 126,  89,   7],

 [  6,  56,  73,  16],

 [ 18,  33,  16, 469]], columns=['1','2','3','4'])
Decesion_entropy=pd.DataFrame([

 [ 75,   5,   2,   0],

 [  8, 125,  96,   2],

 [  3,  82,  49,  17],

 [  6,  28,  24, 478]], columns=['1','2','3','4'])
Logistic=pd.DataFrame([

[ 28,  38,   8,   8],

 [ 62, 104,  39,  26],

 [ 10,  79,  49,  13],

 [ 20,  38,  24, 454]], columns=['1','2','3','4'])
Nural=pd.DataFrame([

[ 49,  17,  15,   1],

 [ 12,  94,  71,  54],

 [ 15,  28,  56,  52],

 [ 19,  21,  11, 485]], columns=['1','2','3','4'])
print("Confusion Matrix: ")

print("")

print("Decision Tree with GINI")

display(Decesion_gini)

print("Accuracy :  74.6")

print()

print()

print("Decision Tree with Entropy")

display(Decesion_entropy)

print("Accuracy :  72.7")

print()

print()

print("Logistic Regression with Multi class Classfication")

display(Logistic)

print("Accuracy :  63.5")

print()

print()

print("Nural Network with Multi Class Classfication")

display(Nural)

print("Accuracy :  68.4")