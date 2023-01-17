# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
%config InlineBackend.figure_format = 'retina'

from matplotlib import style

style.use("seaborn-white")

dat = pd.read_csv('../input/adult.csv')
#Replace the '?' that appear in some of the variables with np.nan

dat.replace('?', np.nan, inplace=True)
import matplotlib.pyplot as plt



def barsize(bars):

    for bar in bars:

        height = bar.get_height()

        ax.text(bar.get_x()+bar.get_width()/2, height+.5, '%d' % int(height), 

                ha='center', va='bottom')



plt.figure(figsize=(6,6))

bars=plt.bar(left=np.arange(dat['income'].nunique()), height=list(dat['income'].value_counts()\

                                                                  .values), align='center')

plt.xticks(np.arange(dat['income'].nunique())/1.00, list(dat['income'].value_counts().index),

           fontweight='bold')

plt.title('Counts of Persons by Income', fontweight='bold', fontsize=16)

ax = plt.gca()

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)

ax.spines['left'].set_visible(False)

ax.set_yticklabels("")

barsize(bars)

plt.figtext(.1, .01, 'Fig. 1: People in the sample making $50000 or less outnumber those  \n'

            'making more by about a little over 3:1.');
import matplotlib.gridspec as gridspec

import seaborn as sns

gs = gridspec.GridSpec(7,2)

plt.figure(figsize=(17,34))

   

#Age by Income Group

ax = plt.subplot(gs[0,0])

ax.hist(dat.loc[dat['income']=="<=50K", 'age'], color="#4286f4")

ax.hist(dat.loc[dat['income']==">50K", 'age'], color="#f45641")

ax.set_title("Distribution of Ages by Income Group", fontsize=16, fontweight='bold')

ax.legend(['<=50K', '>50K'], title="Income Group")

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)

ax.spines['left'].set_visible(False)

ax.set_xlabel("Age")



#Workclass (binned) by Income Group

ax1 = plt.subplot(gs[0,1])

sns.countplot(dat['workclass'], hue=dat['income'], ax=ax1, palette=['#4286f4', '#f45641'])

ax1.legend(loc='upper right', title="Income Group")

ax1.spines['top'].set_visible(False)

ax1.spines['right'].set_visible(False)

ax1.spines['left'].set_visible(False)

ax1.set_xticklabels(['Private', 'State\nGov.', 'Federal\nGov.', 'Self\nEmp\nNot Inc',

                     'Self\nEmp\nInc', 'Local\nGov', 'Without\npay', 'Never\nworked'])

ax1.set_title('Counts of Workclass by Income Group', fontsize=16, fontweight='bold')

ax1.set_xlabel(""), ax1.set_ylabel("")



#Final Weight by Income Group

ax2 = plt.subplot(gs[1,0])

ax2.hist(dat.loc[dat['income']=="<=50K", 'fnlwgt'], color="#4286f4")

ax2.hist(dat.loc[dat['income']==">50K", 'fnlwgt'], color="#f45641")

ax2.set_title("Distribution of Final Weights by Income Group", fontsize=16, fontweight='bold')

ax2.spines['top'].set_visible(False)

ax2.spines['right'].set_visible(False)

ax2.spines['left'].set_visible(False)

ax2.legend(['<=50K', '>50K'], title="Income Group")

ax2.set_xlabel("Final Weight")



#Education by Income Group

ax3 = plt.subplot(gs[1,1])

sns.countplot(dat['education'], hue=dat['income'], ax=ax3, palette=['#4286f4', '#f45641'])

ax3.set_xticklabels(['HS\ngrad', 'Some \ncollege', '7th-\n8th', '10th', 'PhD',

                     'Prof\nschool', 'Masters', '11th', 'Assoc\nacdm', 'Assoc\nvoc',

                     '1st-\n4th', '5th-\n6th', '12th', '9th', 'Pre-\nschool'])

ax3.set_xlabel(""), ax3.set_ylabel("")

ax3.spines['top'].set_visible(False)

ax3.spines['right'].set_visible(False)

ax3.spines['left'].set_visible(False)

ax3.legend(loc='upper right', title="Income Group")

ax3.set_title('Counts of Education by Income Group', fontsize=16, fontweight='bold')



#Education num by Income Group

ax4 = plt.subplot(gs[2,0])

sns.countplot(dat['education.num'], hue=dat['income'], ax=ax4, palette=['#4286f4', '#f45641'])

ax4.set_xlabel('Years of Education'), ax4.set_ylabel("")

ax4.spines['top'].set_visible(False)

ax4.spines['right'].set_visible(False)

ax4.spines['left'].set_visible(False)

ax4.set_title("Counts of Years of Education by Income Group", fontsize=16, fontweight='bold')

ax4.legend(title="Income Group")



#Marital status by Income Group

ax5 = plt.subplot(gs[2,1])

sns.countplot(dat['marital.status'], hue=dat['income'], ax=ax5, palette=['#4286f4', '#f45641'])

ax5.spines['top'].set_visible(False)

ax5.spines['right'].set_visible(False)

ax5.spines['left'].set_visible(False)

ax5.set_ylabel(""), ax5.set_xlabel("")

ax5.set_xticklabels(['Widowed', 'Divorced', 'Separated', 'Never-\nmarried',

                     'Married-\nciv-spouse', 'Married-\nspouse-\nabsent',

                     'Married-\nAF-\nspouse'])

ax5.legend(title="Income Group")

ax5.set_title("Counts of Marital status by Income Group", fontsize=16, fontweight='bold')



#Occupation by Income Group

ax6 = plt.subplot(gs[3,0])

sns.countplot(dat['occupation'], hue=dat['income'], ax=ax6, palette=['#4286f4', '#f45641'])

ax6.spines['top'].set_visible(False)

ax6.spines['right'].set_visible(False)

ax6.spines['left'].set_visible(False)

ax6.set_ylabel(""), ax6.set_xlabel("")

ax6.set_xticklabels(['Exec-\nmgr.', 'Machine-\nop-insp.', 'Prof-\nspecialty',

                     'Other-\nserv.', 'Adm-\nclerical', 'Craft-\nrepair', 

                     'Transport\n-moving', 'Handlers\n-cleaners', 'Sales', 'Farming-\nfishing',

                     'Tech-\nsupport', 'Protective\n-serv.', 'Armed-\nForces', 

                     'Priv-\nhouse-\nserv'])

ax6.legend(title="Income Group")

ax6.set_title("Counts of Occupation by Income Group", fontsize=16, fontweight='bold')



#Relationship by Income Group

ax7 = plt.subplot(gs[3,1])

sns.countplot(dat['relationship'], hue=dat['income'], ax=ax7, palette=['#4286f4', '#f45641'])

ax7.spines['top'].set_visible(False)

ax7.spines['right'].set_visible(False)

ax7.spines['left'].set_visible(False)

ax7.set_ylabel(""), ax7.set_xlabel("")

ax7.legend(title="Income Group")

ax7.set_title("Counts of Relationship Status by Income Group", fontsize=16, fontweight='bold')



#Race by Income Group

ax8 = plt.subplot(gs[4,0])

sns.countplot(dat['race'], hue=dat['income'], ax=ax8, palette=['#4286f4', '#f45641'])

ax8.spines['top'].set_visible(False)

ax8.spines['right'].set_visible(False)

ax8.spines['left'].set_visible(False)

ax8.set_ylabel(""), ax8.set_xlabel("")

ax8.legend(title="Income Group")

ax8.set_title("Counts of Race by Income Group", fontsize=16, fontweight='bold')



#Sex by Income Group

ax9 = plt.subplot(gs[4,1])

sns.countplot(dat['sex'], hue=dat['income'], ax=ax9, palette=['#4286f4', '#f45641'])

ax9.spines['top'].set_visible(False)

ax9.spines['right'].set_visible(False)

ax9.spines['left'].set_visible(False)

ax9.set_ylabel(""), ax9.set_xlabel("")

ax9.legend(title="Income Group")

ax9.set_title("Counts of Gender by Income Group", fontsize=16, fontweight='bold')



#Capital Gain by Income Group

ax10 = plt.subplot(gs[5,0])

ax10.hist(dat.loc[dat['income']=="<=50K", 'capital.gain'], color="#4286f4")

ax10.hist(dat.loc[dat['income']==">50K", 'capital.gain'], color="#f45641")

ax10.spines['top'].set_visible(False)

ax10.spines['right'].set_visible(False)

ax10.spines['left'].set_visible(False)

ax10.set_ylabel(""), ax10.set_xlabel("Capital Gain")

ax10.legend(title="Income Group")

ax10.set_title('Distribution of Capital Gains by Income Group', fontsize=16, fontweight='bold')



#Capital Loss by Income Group

ax11 = plt.subplot(gs[5,1])

ax11.hist(dat.loc[dat['income']=="<=50K", 'capital.loss'], color="#4286f4")

ax11.hist(dat.loc[dat['income']==">50K", 'capital.loss'], color="#f45641")

ax11.spines['top'].set_visible(False)

ax11.spines['right'].set_visible(False)

ax11.spines['left'].set_visible(False)

ax11.set_ylabel(""), ax11.set_xlabel("Capital Loss")

ax11.legend(title="Income Group")

ax11.set_title('Distribution of Capital Loss by Income Group', fontsize=16, fontweight='bold')



#Hours per week by Income Group

ax12 = plt.subplot(gs[6,0])

ax12.hist(dat.loc[dat['income']=="<=50K", 'hours.per.week'], color="#4286f4")

ax12.hist(dat.loc[dat['income']==">50K", 'hours.per.week'], color="#f45641")

ax12.spines['top'].set_visible(False)

ax12.spines['right'].set_visible(False)

ax12.spines['left'].set_visible(False)

ax12.set_ylabel(""), ax12.set_xlabel("Hours Worked")

ax12.legend(title="Income Group")

ax12.set_title('Distribution of Weekly Hours Worked by Income Group', 

               fontsize=16, fontweight='bold')



#Native country by Income Group

ax13 = plt.subplot(gs[6,1])

sns.countplot(y=dat['native.country'], hue=dat['income'], ax=ax13, palette=['#4286f4', 

                                                                            '#f45641'])

ax13.spines['top'].set_visible(False)

ax13.spines['right'].set_visible(False)

ax13.spines['left'].set_visible(False)

ax13.set_ylabel(""), ax13.set_xlabel("")

ax13.legend(title="Income Group")

ax13.set_title("Counts of Native Country by Income Group", fontsize=16, fontweight='bold')



plt.tight_layout();
#Convert 'income' to indicator variable:



income_dummies = pd.get_dummies(dat['income'], prefix='income', drop_first=True)

dat = pd.concat([dat, income_dummies], axis=1)

del dat['income']



#Native.country:



#Convert value from non United States country to 'Not United States'

dat.loc[(dat['native.country']!='United-States') & (dat['native.country'].notnull()), 

        'native.country']='Not United States'



country_dummies = pd.get_dummies(dat['native.country'], prefix='origin_country', 

                                 drop_first=True,

                                dummy_na=True)



dat = pd.concat([dat, country_dummies], axis=1)

del dat['native.country']



#Capital gain/loss: Binned into 0 if capital loss == 0, 1 otherwise. Same for capital gain



dat.loc[dat['capital.loss']>0, 'capital.loss'] = 1

dat.loc[dat['capital.gain']>0, 'capital.gain'] = 1



#Sex:



sex_dummies = pd.get_dummies(dat['sex'], prefix='gender', drop_first=True)

dat = pd.concat([dat, sex_dummies], axis=1)

del dat['sex']



#Race: Binned into White/Non-White:



race_dict = {'Black': 'non_White', 'Asian-Pac-Islander': 'non_White',

             'Other': 'non_White', 'Amer-Indian-Eskimo': 'non_White'}



race_dummies = pd.get_dummies(dat['race'].replace(race_dict.keys(), race_dict.values()),

                              prefix='race', drop_first=True)

dat = pd.concat([dat, race_dummies], axis=1)

del dat['race']



#Occupation: Armed Forces binned with Protective-Service bc only 9 in the military. 



occupy_dict = {'Armed-Forces': 'Protective-serv-Military', 'Protective-serv': 

               'Protective-serv-Military'}



occupy_dummies = pd.get_dummies(dat['occupation'].replace(occupy_dict.keys(),

                                                          occupy_dict.values()),

                                                          prefix='occupation', drop_first=True,

                                                          dummy_na=True)

dat = pd.concat([dat, occupy_dummies], axis=1)

del dat['occupation']



#Marital Status: married subgroups binned into one group 'married':



married_dict = {'Married-civ-spouse': 'Married', 'Married-spouse-absent': 'Married',

                'Married-AF-spouse': 'Married'}



marital_dummies = pd.get_dummies(dat['marital.status'].replace(married_dict.keys(),

                                                               married_dict.values()),

                                                               prefix='marital_status',

                                                               drop_first=True)

dat = pd.concat([dat, marital_dummies], axis=1)

del dat['marital.status']



#education: binned



education_dict = {'1st-4th': 'Grade-school', '5th-6th': 'Grade-school', '7th-8th': 

                  'Junior-high', '9th': 'HS-nongrad', '10th': 'HS-nongrad', 

                  '11th': 'HS-nongrad', '12th': 'HS-nongrad', 'Masters': 

                  'Graduate', 'Doctorate': 'Graduate', 'Preschool': 'Grade-school'}



educ_dummies = pd.get_dummies(dat.education.replace(education_dict.keys(), 

                                                    education_dict.values()),

                                                    prefix='education',

                                                    drop_first=True)

                              

dat = pd.concat([dat, educ_dummies], axis=1)

del dat['education']



#workclass:



#Those who have a workclass of 'never worked' or 'without pay' will be dropped as we want to

#focus our attention on wage earners:



dat.drop(dat.loc[(dat.workclass=='Without-pay') | (dat.workclass=='Never-worked'), :].index,

        inplace=True)



class_dict = {'Local-gov': 'Government', 'State-gov': 'Government', 'Federal-gov': 'Government',

              'Self-emp-not-inc': 'Self-employed', 'Self-emp-inc': 'Self-employed'}



class_dummies= pd.get_dummies(dat.workclass.replace(class_dict.keys(), class_dict.values()),

                              prefix='workclass', drop_first=True, dummy_na=True)



dat = pd.concat([dat, class_dummies], axis=1)

del dat['workclass']



#relationship: not sure what this variable is about but with just a few levels, I will 

#create a set of dummies for all of them



relate_dummies = pd.get_dummies(dat.relationship, prefix='relationship', drop_first=True)



dat = pd.concat([dat, relate_dummies], axis=1)

del dat['relationship']





#Age



#applying a log transformation on age to maintain its interpretability and 

#make variable's scale closer to values of indicator variables.



dat['age'] = np.log10(dat['age'])



#fnlwgt: not quite sure the purpose of this variable with this data but holding

#on to it at least until we build baseline. Going to transform it to log scale



dat['fnlwgt'] = np.log10(dat['fnlwgt'])



#hours worked will be binned. 35-40hrs will be 'full-time'; <35 will be part-time;

#>40 will be '40+hrs'



dat['hours.worked'] = np.nan

dat.loc[(dat['hours.per.week']>=35) | (dat['hours.per.week']<=40), 'hours.worked'] = 'Full_time'

dat.loc[dat['hours.per.week']<35, 'hours.worked'] = 'Part_time'

dat.loc[dat['hours.per.week']>40, 'hours.worked'] = '40+hrs'



hours_dummies = pd.get_dummies(dat['hours.worked'], prefix='WklyHrs', drop_first=True)



dat = pd.concat([dat, hours_dummies], axis=1)



del dat['hours.per.week']

del dat['hours.worked']



#Education num will be binned in 4 year increments



educ_dict = {1: '1-4', 2: '1-4', 3: '1-4', 4: '1-4', 5: '5-8', 6: '5-8',

             7: '5-8', 8: '5-8', 9: '9-12', 10: '9-12', 11: '9-12', 12: '9-12',

             13: '13-16', 14: '13-16', 15: '13-16', 16: '13-16'}



educ_num = pd.get_dummies(dat['education.num'].replace(educ_dict.keys(), educ_dict.values()),

                          prefix='YrsEduc', drop_first=False)



dat = pd.concat([dat, educ_num], axis=1)



del dat['education.num']



#Splitting data into a training and test set (test set = 33% of data)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(dat.loc[:, dat.columns != 'income_>50K'],

                                                    dat.loc[:, 'income_>50K'], test_size=.33,

                                                    random_state=1234)





from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score



scores = cross_val_score(LogisticRegression(), X_train, y_train, cv=10, scoring='accuracy')



print(scores)

print(scores.mean())
base_log = LogisticRegression()

base_log.fit(X_train, y_train)

base_preds = base_log.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



print(accuracy_score(y_test, base_preds))
#Confusion matrix:



print(confusion_matrix(y_test, base_preds))
#Classification report:



print(classification_report(y_test, base_preds))