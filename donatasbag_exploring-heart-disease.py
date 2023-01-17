import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report



from sklearn.neural_network import MLPClassifier        # Multy level Perseptatthron Classifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier      

                                                        # Random Forest Classifier, Gradient Boosting

from sklearn.svm import SVC                             # Support Vector Classifier

from sklearn.linear_model import LogisticRegression     # Logistic Regression

from sklearn.tree import DecisionTreeClassifier         # Decission Tree

from sklearn import svm





%matplotlib inline

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/heart.csv')

df.tail(4)
df.info()
f,ax = plt.subplots(figsize=(18, 6))

sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()



sns.set(style="darkgrid")

plt.title('Heart disease in total')

sns.countplot(x="target", data=df, facecolor=(1, 0, 1, 0),linewidth=5, edgecolor=sns.color_palette("dark", 3))

plt.xlabel("Target ({} for HD negative, {} for HD positive)".format(0, 1))



No_Disease, Disease = (len(df[df.target == 0]), len(df[df.target == 1]))

len_heart = df.shape[0]



print("Patients which havn't heart disease: {:.0f}".format(No_Disease))

print("Patients which have heart disease: {:.0f}".format(Disease))

print()

print("Patients which havn't heart disease: {:.0f}%".format(No_Disease/(len_heart)*100))

print("Patients which have heart disease: {:.0f}%".format(Disease/(len_heart)*100))

col = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

for item in col:

    pd.crosstab(df[item], df.target).plot.bar(figsize=(15, 5))

    plt.title("{} with target".format(str(item)))

    plt.legend(["dont have disease", "have disease"])

    plt.ylabel("Count")

plt.show()
print("                    Patients by age       ") 

print()

# less than 50 years

less, less_notdiseased, less_diseased = len(df[df.age <= 50]), len(df[(df.target == 0) & (df.age <= 50)]), len(df[(df.target == 1) & (df.age <= 50)])

print("Out of %s younger than 50 years of age patients - %s dont have heart disease and - %s do have" %(less, less_notdiseased, less_diseased))

print("Dont have heart disease: {:.0f}%".format(less_notdiseased/(less)*100))  

print("Have heart disease: {:.0f}%".format(less_diseased/(less)*100)) 



line_labels = 'Dont have heart disease', 'Have heart disease'



sizes1 = [29, 66]

 

print()

# More than 50

more, more_notdiseased,more_diseased = len(df[df.age > 50]), len(df[(df.target == 0) & (df.age > 50)]), len(df[(df.target == 1) & (df.age > 50)])

print("Out of %s older than 50 years of age patients  - %s dont have heart disease and - %s do have" %(more, more_notdiseased,more_diseased))

print("Dont have heart disease: {:.0f}%".format(more_notdiseased/(more)*100))  

print("Have heart disease: {:.0f}%".format(more_diseased/(more)*100)) 



sizes2 = [109,99]



fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 6))

fig.patch.set_facecolor('darkgrey')

l1 = ax1.pie(sizes1, autopct='%1.0f%%',

        shadow=True, startangle=90)

l2 = ax2.pie(sizes2, autopct='%1.0f%%',

        shadow=True, explode=(0, 0))

ax1.set_title('Younger than 50 years')

ax2.set_title('Older than 50 years')

fig.legend([l1, l2],    

           labels=line_labels, 

           loc="center left",  

           borderaxespad=0.1,   

           title="Diagnosis" 

           )



plt.subplots_adjust(right=0.9)

plt.show()



mean = df.age.mean()

max = df.age.max()

min = df.age.min()

tot, diag = less+more, ((less_diseased + more_diseased)/(less + more)) *100



print("Average patiens age is %s. Oldes is - %s years of age and the youngest - %s  " %(round(mean), max, min))

print("To %s %% out of all %s patiens heart disease  was diagnosed" %(round(diag), tot))

print ("2 of 3 patiens are older than 50 years, but only 48 % of them heart disease diagnosis is positive")

print ("2 of 3 patiens younger than 50 years, heart disease diagnosis is positive")
print("                    Patients by gender       ") 

print()

# Male

male, male_notdiseased, male_diseased = len(df[df.sex == 1]), len(df[(df.sex == 1) & (df.target == 0)]), len(df[(df.sex == 1) & (df.target == 1)])



print("Out of %s male patients - %s dont have heart disease and - %s do have " %(male, male_notdiseased, male_diseased))

print("Dont have heart disease: {:.0f}%".format(male_notdiseased/(male)*100))  

print("Have heart disease: {:.0f}%".format(male_diseased/(male)*100)) 



line_labels = 'Dont have disease', 'Have disease'



sizes1 = [114,93]

 

print()

# Female

female, female_notdiseased, female_diseased = len(df[df.sex == 0]), len(df[(df.sex == 0) & (df.target == 0)]), len(df[(df.sex == 0) & (df.target == 1)])



print("Out of %s female patients - %s dont have heart disease and - %s do have" %(female, female_notdiseased, female_diseased))

print("Dont have heart disease: {:.0f}%".format(female_notdiseased/(female)*100))  

print("Have heart disease: {:.0f}%".format(female_diseased/(female)*100)) 



sizes2 = [24, 72]



fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 6))

fig.patch.set_facecolor('darkgrey')

l1 = ax1.pie(sizes1, autopct='%1.0f%%',

        shadow=True, startangle=90)

l2 = ax2.pie(sizes2, autopct='%1.0f%%',

        shadow=True, explode=(0, 0))

ax1.set_title('Male')

ax2.set_title('Female')

fig.legend([l1, l2],    

           labels=line_labels, 

           loc="center left",  

           borderaxespad=0.1,   

           title="Diagnosis" 

           )



plt.subplots_adjust(right=0.9)

plt.show()



print ("2 of 3 patiens are male, but to 55% of them heart disease diagnosis is rejected")

print ("To 3 of 4 female patients heart disease diagnosis is positive")
print("                    Patients by chest pain types       ") 

print()

# Dont have disease by chest pain types

targ_1n, cp_1n, cp_2n, cp_3n, cp_4n = len(df[df.target == 0]), len(df[(df.target == 0) & (df.cp == 0)]), len(df[(df.target == 0) & (df.cp == 1)]), len(df[(df.target == 0) & (df.cp == 2)]), len(df[(df.target == 0) & (df.cp == 3)])



print("%s patients, that dont have heart disease have:  %s - first, %s - second, %s - third and %s - fourth chest type pains" %(targ_1n, cp_1n, cp_2n, cp_3n, cp_4n))

print("1 type chest pain: {:.0f}%".format(cp_1n/(targ_1n)*100))

print("2 type chest pain: {:.0f}%".format(cp_2n/(targ_1n)*100))

print("3 type chest pain: {:.0f}%".format(cp_3n/(targ_1n)*100))

print("4 type chest pain: {:.0f}%".format(cp_4n/(targ_1n)*100))



line_labels = '1 type', '2 type', '3 type', '4 type'



sizes1 = [104, 9, 18, 7]

 

print()

# Have disease by chest pain types

targ_1d, cp_1d, cp_2d, cp_3d, cp_4d = len(df[df.target == 1]), len(df[(df.target == 1) & (df.cp == 0)]), len(df[(df.target == 1) & (df.cp == 1)]), len(df[(df.target == 1) & (df.cp == 2)]), len(df[(df.target == 1) & (df.cp == 3)])



print("%s patients, that do have heart disease have:  %s - first, %s - second, %s - third and %s - fourth chest type pains" %(targ_1d, cp_1d, cp_2d, cp_3d, cp_4d))

print("1 type chest pain: {:.0f}%".format(cp_1d/(targ_1d)*100))

print("2 type chest pain: {:.0f}%".format(cp_2d/(targ_1d)*100))

print("3 type chest pain: {:.0f}%".format(cp_3d/(targ_1d)*100))

print("4 type chest pain: {:.0f}%".format(cp_4d/(targ_1d)*100))



sizes2 = [39,41,69,16]



fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 6))

fig.patch.set_facecolor('darkgrey')

l1 = ax1.pie(sizes1, autopct='%1.0f%%',

        shadow=True, startangle=90)

l2 = ax2.pie(sizes2, autopct='%1.0f%%',

        shadow=True, explode=(0, 0, 0, 0))

ax1.set_title('Dont have disease')

ax2.set_title('Have disease')

fig.legend([l1, l2],    

           labels=line_labels, 

           loc="center left",  

           borderaxespad=0.1,   

           title="Chest pain type" 

           )



plt.subplots_adjust(right=0.9)

plt.show()

print("3 of 4 healthy patients have 1 type chest pain and that type of chets pain have only 1 of 4 unhealthy patients")

print("Each of 2, 3 and 4 type of chest pain was higher for patiens that have heart disease compared to those who havent ")
print("                    Patients by resting blood pressure       ") 

print()

# Dont have disease by resting blood pressure

ntarg, nnormal, nelevated, nhypertension_stage1, nhypertension_stage2, nhypertensive_crisis  = len(df[df.target == 0]), len(df[(df.target == 0) & (df.trestbps < 120)]),len(df[(df.target == 0) & (df.trestbps.between(120, 129, inclusive=True))]) ,len(df[(df.target == 0) & (df.trestbps.between(130, 139, inclusive=True))]),len(df[(df.target == 0) & (df.trestbps.between(140, 180, inclusive=True))]), len(df[(df.target == 0) & (df.trestbps >= 181)])

print("%s patients, that dont have heart disease have:  %s - normal, %s - elevated, %s - high blood pressure (hypertension) stage 1, %s - high blood pressure (hypertension) stage 2, %s - hypertensive crisis" %(ntarg, nnormal, nelevated, nhypertension_stage1, nhypertension_stage2, nhypertensive_crisis))

print(" --- Normal resting blood pressure(less than 120): {:.0f}%".format(nnormal/(ntarg)*100))

print(" --- Elevated resting blood pressure(120 - 129): {:.0f}%".format(nelevated/(ntarg)*100))

print(" --- High blood pressure (hypertension) stage 1 (130 - 139): {:.0f}%".format(nhypertension_stage1/(ntarg)*100))

print(" --- High blood pressure (hypertension) stage 2 (140 or higher): {:.0f}%".format(nhypertension_stage2/(ntarg)*100))

print(" --- hypertensive crisis (higher than 180): {:.0f}%".format(nhypertensive_crisis/(ntarg)*100))



line_labels = 'Normal', 'Elevated', 'Hypertension stage 1', 'Hypertension stage 2', 'Hypertensive crisis'



sizes1 = [23, 35, 27, 51, 2]

print()

# Have disease by resting blood pressure

dtarg, dnormal, delevated, dhypertension_stage1, dhypertension_stage2, dhypertensive_crisis  = len(df[df.target == 1]), len(df[(df.target == 1) & (df.trestbps < 120)]),len(df[(df.target == 1) & (df.trestbps.between(120, 129, inclusive=True))]) ,len(df[(df.target == 1) & (df.trestbps.between(130, 139, inclusive=True))]),len(df[(df.target == 1) & (df.trestbps.between(140, 180, inclusive=True))]), len(df[(df.target == 1) & (df.trestbps >= 181)])

print("%s patients, that dont have heart disease have:  %s - normal, %s - elevated, %s - high blood pressure (hypertension) stage 1, %s - high blood pressure (hypertension) stage 2, %s - hypertensive crisis" %(dtarg, dnormal, delevated, dhypertension_stage1, dhypertension_stage2, dhypertensive_crisis))

print(" --- Normal resting blood pressure(less than 120): {:.0f}%".format(dnormal/(dtarg)*100))

print(" --- Elevated resting blood pressure(120 - 129): {:.0f}%".format(delevated/(dtarg)*100))

print(" --- High blood pressure (hypertension) stage 1 (130 - 139): {:.0f}%".format(dhypertension_stage1/(dtarg)*100))

print(" --- High blood pressure (hypertension) stage 2 (140 or higher): {:.0f}%".format(dhypertension_stage2/(dtarg)*100))

print(" --- hypertensive crisis (higher than 180): {:.0f}%".format(dhypertensive_crisis/(dtarg)*100))



sizes2 = [37, 40, 44, 44, 0]

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 6))

fig.patch.set_facecolor('darkgrey')

l1 = ax1.pie(sizes1, autopct='%1.0f%%',

        shadow=True, startangle=90)

l2 = ax2.pie(sizes2, autopct='%1.0f%%',

        shadow=True, explode=(0, 0, 0, 0, 0))

ax1.set_title('Dont have disease')

ax2.set_title('Have disease')

fig.legend([l1, l2],    

           labels=line_labels, 

           loc="center left",  

           borderaxespad=0.1,   

           title="Resting blood pressure" 

           )



plt.subplots_adjust(right=0.9)

plt.show()

print("To - 83 % of patients, that dont have heart disease, resting blood pressure is higher than normal and  - 78 % who have heart disease")

print("No strong deviations regarding resting blood pressure among patients who have and dont have heart disease noticed. Even thou high blood pressure can be assigned to diseases non related to heart disease, not normal resting blood pressure remain one of the main risk factors")
print("                    Patients by serum cholestoral in mg/dL       ") 

print()

# Dont have disease by serum cholestoral in mg/dl

ntarg_1, nchol_good, nchol_borderline, nchol_bad = len(df[df.target == 0]), len(df[(df.target == 0) & (df.chol < 200)]), len(df[(df.target == 0) & (df.chol.between(200, 239, inclusive=True))]), len(df[(df.target == 0) & (df.chol >= 240)])

print("%s patients that dont have heart disease have: %s - normal cholestoral, %s - borderline cholestoral and %s - high cholestoral" %(ntarg_1, nchol_good, nchol_borderline, nchol_bad))

print("Normal cholestoral (less 200 mg/dL): {:.0f}%".format(nchol_good/(ntarg_1)*100))

print("Borderline cholestoral(between 200-240 mg/dL): {:.0f}%".format(nchol_borderline/(ntarg_1)*100))  

print("High cholestoral (above 240 mh/dL): {:.0f}%".format(nchol_bad/(ntarg_1)*100))  

print()

line_labels = 'Normal', 'Borderline', 'High'

sizes1 = [20, 39, 79]



# Have disease by serum cholestoral in mg/dl

dtarg_1, dchol_good, dchol_borderline, dchol_bad = len(df[df.target == 1]), len(df[(df.target == 1) & (df.chol < 200)]), len(df[(df.target == 1) & (df.chol.between(200, 239, inclusive=True))]), len(df[(df.target == 1) & (df.chol >= 240)])

print("%s patients that have heart disease have: %s  - normal cholestoral, %s - borderline cholestoral and %s - high cholestoral" %(dtarg_1, dchol_good, dchol_borderline, dchol_bad))

print("Normal cholestoral (less 200 mg/dL): {:.0f}%".format(dchol_good/(dtarg_1)*100))

print("Borderline cholestoral(between 200-240 mg/dL): {:.0f}%".format(dchol_borderline/(dtarg_1)*100))  

print("High cholestoral (above 240 mh/dL): {:.0f}%".format(dchol_bad/(dtarg_1)*100))  



sizes2 = [30, 59, 76]



fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 6))

fig.patch.set_facecolor('darkgrey')

l1 = ax1.pie(sizes1, autopct='%1.0f%%',

        shadow=True, startangle=90)

l2 = ax2.pie(sizes2, autopct='%1.0f%%',

        shadow=True, explode=(0, 0, 0))

ax1.set_title('Dont have disease')

ax2.set_title('Have disease')

fig.legend([l1, l2],    

           labels=line_labels, 

           loc="center left",  

           borderaxespad=0.1,   

           title="Serum cholestoral" 

           )



plt.subplots_adjust(right=0.9)

plt.show()

print("For - 86 % of patients that have no heart disease serum cholestoral in mg/dl is higher than normal and - 82 % of patients that have heart disease")

print("No strong deviations regarding serum cholesteral among patients noticed. Even thou high serum cholestoral can be assigned to diseases non related to heart disease, higher normal serum cholestoral in mg/dl remain one of the main risk factors")
print("                    Patients by fasting blood sugar       ") 

print()

# Dont have disease by fasting blood sugar

ntarg_1, nfbs0, nfbs1 = len(df[df.target == 0]), len(df[(df.target == 0) & (df.fbs == 1)]), len(df[(df.target == 0) & (df.fbs == 0)])



print("Out of %s patients that dont have heart disease - %s have high fasting blood sugar - %s have low fasting blood sugar" %(ntarg_1, nfbs0, nfbs1))

print("Have high fasting blood sugar: {:.0f}%".format(nfbs0/(ntarg_1)*100))  

print("Have low fasting blood sugar: {:.0f}%".format(nfbs1/(ntarg_1)*100)) 

print()

line_labels = 'High fasting blood sugar', 'Low fasting blood sugar'

sizes1 = [22, 116]



# Have disease by fasting blood sugar

dtarg_1, dfbs0, dfbs1 = len(df[df.target == 1]), len(df[(df.target == 1) & (df.fbs == 1)]), len(df[(df.target == 1) & (df.fbs == 0)])



print("Out of %s patients that dont have heart disease - %s have high fasting blood sugar - %s have low fasting blood sugar" %(dtarg_1, dfbs0, dfbs1))

print("Have high fasting blood sugar: {:.0f}%".format(dfbs0/(dtarg_1)*100))  

print("Have low fasting blood sugar: {:.0f}%".format(dfbs1/(dtarg_1)*100)) 

print()



sizes2 = [23, 142]



fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 6))

fig.patch.set_facecolor('darkgrey')

l1 = ax1.pie(sizes1, autopct='%1.0f%%',

        shadow=True, startangle=90)

l2 = ax2.pie(sizes2, autopct='%1.0f%%',

        shadow=True, explode=(0, 0))

ax1.set_title('Dont have disease')

ax2.set_title('Have disease')

fig.legend([l1, l2],    

           labels=line_labels, 

           loc="center left",  

           borderaxespad=0.1,   

           title="Fasting blood sugar" 

           )



plt.subplots_adjust(right=0.9)

plt.show()

print()

print("Both groups of patients have obviously decreased level of fasting blood sugar. Fasting blood sugar is serious risk factor for human health in general")
print("                    Patients by electrocardiogramic results       ") 

print()

# Not diseased electrocardiogramic result

ntarg_1, nrestecg0, nrestecg1, nrestecg2 = len(df[df.target == 0]), len(df[(df.target == 0) & (df.restecg == 0)]), len(df[(df.target == 0) & (df.restecg == 1)]), len(df[(df.target == 0) & (df.restecg == 2)])



print("%s patients that have no heart disease have: %s - electrocardiogramic result 0 , %s - electrocardiogramic result 1, %s - electrocardiogramic result 2" %(ntarg_1, nrestecg0, nrestecg1, nrestecg2))

print("Electrocardiogramic result - 0: {:.0f}%".format(nrestecg0/(ntarg_1)*100))

print("Electrocardiogramic result - 1: {:.0f}%".format(nrestecg1/(ntarg_1)*100))

print("Electrocardiogramic result - 2: {:.0f}%".format(nrestecg2/(ntarg_1)*100))

print()

line_labels = 'Result 0', 'Result 1', 'Result 2'

sizes1 = [79,56,3]



# Diseased electrocardiogramic result

dtarg_1, drestecg0, drestecg1, drestecg2 = len(df[df.target == 1]), len(df[(df.target == 1) & (df.restecg == 0)]), len(df[(df.target == 1) & (df.restecg == 1)]), len(df[(df.target == 1) & (df.restecg == 2)])



print("%s patients that have no heart disease have: %s - electrocardiogramic result 0 , %s - electrocardiogramic result 1, %s - electrocardiogramic result 2" %(dtarg_1, drestecg0, drestecg1, drestecg2))

print("Electrocardiogramic result - 0: {:.0f}%".format(drestecg0/(dtarg_1)*100))

print("Electrocardiogramic result - 1: {:.0f}%".format(drestecg1/(dtarg_1)*100))

print("Electrocardiogramic result - 2: {:.0f}%".format(drestecg2/(dtarg_1)*100))



sizes2=[68, 96, 1]



fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 6))

fig.patch.set_facecolor('darkgrey')

l1 = ax1.pie(sizes1, autopct='%1.0f%%',

        shadow=True, startangle=90)

l2 = ax2.pie(sizes2, autopct='%1.0f%%',

        shadow=True, explode=(0, 0, 0))

ax1.set_title('Dont have disease')

ax2.set_title('Have disease')

fig.legend([l1, l2],    

           labels=line_labels, 

           loc="center left",  

           borderaxespad=0.1,   

           title="Electrocardiogramic result" 

           )



plt.subplots_adjust(right=0.9)

plt.show()

print("Wery strong correlation between Result 0 and Result 1. The same but reversed percentage shows electrocardiogramic result accuracy ")
print("                    Patients by maximum heart rate acheaved       ") 

print()

# Dont have disease by maximum heart rate acheaved

ntarg_1, nthalach_min, nthalach_max = len(df[df.target == 0]), len(df[(df.target == 0) & (df.thalach <= 150)]), len(df[(df.target == 0) & (df.thalach > 150)])



print("%s patients that have no heart disease have: %s - lower than 150 maximum heart rate acheaved and %s - higher" %(ntarg_1, nthalach_min, nthalach_max))

print("Heart rate less than 150: {:.0f}%".format(nthalach_min/(ntarg_1)*100))

print("Heart rate more than 150: {:.0f}%".format(nthalach_max/(ntarg_1)*100))  

print()

line_labels = 'Less than 150', 'More than 150'

sizes1 = [94, 44]

  

# Diseased by maximum heart rate acheaved

dtarg_1, dthalach_min, dthalach_max = len(df[df.target == 1]), len(df[(df.target == 1) & (df.thalach <= 150)]), len(df[(df.target == 1) & (df.thalach > 150)])



print("%s patients that have heart disease have: %s - lower than 150 maximum heart rate acheaved and %s - higher" %(dtarg_1, dthalach_min, dthalach_max))

print("Heart rate less than 150: {:.0f}%".format(dthalach_min/(dtarg_1)*100))

print("Heart rate more than 150: {:.0f}%".format(dthalach_max/(dtarg_1)*100))  



sizes2 = [45,120]



fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 6))

fig.patch.set_facecolor('darkgrey')

l1 = ax1.pie(sizes1, autopct='%1.0f%%',

        shadow=True, startangle=90)

l2 = ax2.pie(sizes2, autopct='%1.0f%%',

        shadow=True, explode=(0, 0))

ax1.set_title('Dont have disease')

ax2.set_title('Have disease')

fig.legend([l1, l2],    

           labels=line_labels, 

           loc="center left",  

           borderaxespad=0.1,   

           title="Heart rate acheaved" 

           )



plt.subplots_adjust(right=0.9)

plt.show()

print("Similary as electrocardiogramic results, wery strong reverse correlation betwean patients with disease and the ones dont have it")

print("3/4 patients with heart disease heart rate exceeded 150 during phisical activity")
print("                    Patients by exercise induced angina       ") 

print()

# Dont have disease by exercise induced angina 

ntarg_1, nexang_min, nexang_max = len(df[df.target == 0]), len(df[(df.target == 0) & (df.exang == 0)]), len(df[(df.target == 0) & (df.exang == 1)])



print("%s patients that dont have heart disease have:  %s - exercise induced angina and %s - not exercise induced angina" %(ntarg_1, nexang_min, nexang_max))

print("exercise induced angina: {:.0f}%".format(nexang_min/(ntarg_1)*100))

print("exercise not induced angina: {:.0f}%".format(nexang_max/(ntarg_1)*100))  

print()

line_labels = 'Yes', 'No'

sizes1 = [62,76]



# Have disease by exercise induced angina 

dtarg_1, dexang_min, dexang_max = len(df[df.target == 1]), len(df[(df.target == 1) & (df.exang == 0)]), len(df[(df.target == 1) & (df.exang == 1)])



print("%s patients that dont have heart disease have:  %s - exercise induced angina and %s - not exercise induced angina" %(dtarg_1, dexang_min, dexang_max))

print("Exercise induced angina: {:.0f}%".format(dexang_min/(dtarg_1)*100))

print("Exercise not induced angina: {:.0f}%".format(dexang_max/(dtarg_1)*100))  

print()



sizes2 = [142,23]



fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 6))

fig.patch.set_facecolor('darkgrey')

l1 = ax1.pie(sizes1, autopct='%1.0f%%',

        shadow=True, startangle=90)

l2 = ax2.pie(sizes2, autopct='%1.0f%%',

        shadow=True, explode=(0, 0))

ax1.set_title('Dont have disease')

ax2.set_title('Have disease')

fig.legend([l1, l2],    

           labels=line_labels, 

           loc="center left",  

           borderaxespad=0.1,   

           title="Induced angina by excersise" 

           )



plt.subplots_adjust(right=0.9)

plt.show()

print("Angina by exercie is significantly over 86% induced among patients that have heart disease")
print("                    Patients by oldpeak       ") 

print()

# Dont have disease by oldpeak = ST depression induced by exercise relative to rest

ntarg_1, noldpeak_1, noldpeak_2, noldpeak_3, noldpeak_4 = len(df[df.target == 0]), len(df[(df.target == 0) & (df.oldpeak < 1)]), len(df[(df.target == 0) & (df.oldpeak.between(1,2, inclusive=True))]), len(df[(df.target == 0) & (df.oldpeak.between(2,3, inclusive=False))]), len(df[(df.target == 0) & (df.oldpeak >= 3)])





print("%s patients that dont have disease have: %s - ST depression induced by exercise relative to rest less 1, %s - between 1-2, %s - between 2-3, %s - more 3" %(ntarg_1, noldpeak_1, noldpeak_2, noldpeak_3, noldpeak_4))

print("ST depression induced by exercise relative to rest less 1: {:.0f}%".format(noldpeak_1/(ntarg_1)*100)) 

print("ST depression induced by exercise relative to rest between 1-2: {:.0f}%".format(noldpeak_2/(ntarg_1)*100))  

print("ST depression induced by exercise relative to rest between 2-3: {:.0f}%".format(noldpeak_3/(ntarg_1)*100))

print("ST depression induced by exercise relative to rest more 3: {:.0f}%".format(noldpeak_4/(ntarg_1)*100))

print()

line_labels = 'A type. less 1', 'B type. between 1-2', 'C type. between 2-3', 'D type. more 3'

sizes1 = [46, 49, 21, 22]



# Have disease by oldpeak = ST depression induced by exercise relative to rest

dtarg_1, doldpeak_1, doldpeak_2, doldpeak_3, doldpeak_4 = len(df[df.target == 1]), len(df[(df.target == 1) & (df.oldpeak < 1)]), len(df[(df.target == 1) & (df.oldpeak.between(1,2, inclusive=True))]), len(df[(df.target == 1) & (df.oldpeak.between(2,3, inclusive=False))]), len(df[(df.target == 1) & (df.oldpeak >= 3)])





print("%s patients that have disease have: %s - ST depression induced by exercise relative to rest less 1, %s - between 1-2, %s - between 2-3, %s - more 3" %(dtarg_1, doldpeak_1, doldpeak_2, doldpeak_3, doldpeak_4))

print("ST depression induced by exercise relative to rest less 1: {:.0f}%".format(doldpeak_1/(dtarg_1)*100)) 

print("ST depression induced by exercise relative to rest between 1-2: {:.0f}%".format(doldpeak_2/(dtarg_1)*100))  

print("ST depression induced by exercise relative to rest between 2-3: {:.0f}%".format(doldpeak_3/(dtarg_1)*100))

print("ST depression induced by exercise relative to rest more 3: {:.0f}%".format(doldpeak_4/(dtarg_1)*100))



sizes2 = [120, 38, 4, 3]



fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 6))

fig.patch.set_facecolor('darkgrey')

l1 = ax1.pie(sizes1, autopct='%1.0f%%',

        shadow=True, startangle=90)

l2 = ax2.pie(sizes2, autopct='%1.0f%%',

        shadow=True, explode=(0, 0, 0, 0))

ax1.set_title('Dont have disease')

ax2.set_title('Have disease')

fig.legend([l1, l2],    

           labels=line_labels, 

           loc="center left",  

           borderaxespad=0.1,   

           title="ST depression" 

           )



plt.subplots_adjust(right=0.9)

plt.show()

print("Patients, that dont have heart disease are more likely to have B, C and D type ST depression induced by exercise relative to rest, compared to patients, that have heart disease")

print("Patients, that have heart disease are more likely to have A type of ST depression induced by exercise relative to rest")

print("Almost 3 of 4 patients, that have heart disease have A type of ST depression induced by exercise relative to rest. And A type is diagnosed only to 1 of 3 patients , that dont have heart disease")
print("                    Patients by the slope of the peak exercise ST segment       ") 

print()

# Dont have disease by the slope of the peak exercise ST segment 

ntarg_1, nslope_0, nslope_1, nslope_2 = len(df[df.target == 0]), len(df[(df.target == 0) & (df.slope == 0)]), len(df[(df.target == 0) & (df.slope == 1)]), len(df[(df.target == 0) & (df.slope == 2)])





print("%s patients that dont have heart disease have:  %s - the slope 0 of the peak exercise ST segment, slope 1 - %s, slope 2 - %s" %(ntarg_1, nslope_0, nslope_1, nslope_2))

print("Slope 0 of the peak exercise ST segment: {:.0f}%".format(nslope_0/(ntarg_1)*100)) 

print("Slope 1 of the peak exercise ST segment: {:.0f}%".format(nslope_1/(ntarg_1)*100))  

print("Slope 2 of the peak exercise ST segment: {:.0f}%".format(nslope_2/(ntarg_1)*100))

print()



line_labels = 'Slope 0', 'Slope 1', 'Slope 2'

sizes1 = [12, 91, 35]



# Have disease by the slope of the peak exercise ST segment 

dtarg_1, dslope_0, dslope_1, dslope_2 = len(df[df.target == 1]), len(df[(df.target == 1) & (df.slope == 0)]), len(df[(df.target == 1) & (df.slope == 1)]), len(df[(df.target == 1) & (df.slope == 2)])

print("%s patients that have heart disease have:  %s - the slope 0 of the peak exercise ST segment, slope 1 - %s, slope 2 - %s" %(dtarg_1, dslope_0, dslope_1, dslope_2))

print("Slope 0 of the peak exercise ST segment: {:.0f}%".format(dslope_0/(dtarg_1)*100)) 

print("Slope 1 of the peak exercise ST segment: {:.0f}%".format(dslope_1/(dtarg_1)*100))  

print("Slope 2 of the peak exercise ST segment: {:.0f}%".format(dslope_2/(dtarg_1)*100))



sizes2 = [9, 49, 107]



fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 6))

fig.patch.set_facecolor('darkgrey')

l1 = ax1.pie(sizes1, autopct='%1.0f%%',

        shadow=True, startangle=90)

l2 = ax2.pie(sizes2, autopct='%1.0f%%',

        shadow=True, explode=(0, 0, 0))

ax1.set_title('Dont have disease')

ax2.set_title('Have disease')

fig.legend([l1, l2],    

           labels=line_labels, 

           loc="center left",  

           borderaxespad=0.1,   

           title="Slope" 

           )



plt.subplots_adjust(right=0.9)

plt.show()

print("2 of 3 patients that dont have heart disease have slope 0")

print("2 of 3 patients that have heart disease have slope 2")
print("                    Patients by number of major vessels (0-3) colored by flourosopy       ") 

print()

# Dont have disease by number of major vessels (0-3) colored by flourosopy

ntarg_1, nca_0, nca_1, nca_2, nca_3 = len(df[df.target == 0]), len(df[(df.target == 0) & (df.ca == 0)]), len(df[(df.target == 0) & (df.ca == 1)]), len(df[(df.target == 0) & (df.ca == 2)]), len(df[(df.target == 0) & (df.ca >= 3)])



print("Out of %s patients, that dont have heart disease have: %s - number 0 of major vessels colored by flourosopy, %s - number 1, %s - number 2, %s - number 3" %(ntarg_1, nca_0, nca_1, nca_2, nca_3))

print("Number 0 of major vessels colored by flourosopy: {:.0f}%".format(nca_0/(ntarg_1)*100)) 

print("Number 1 of major vessels colored by flourosopy: {:.0f}%".format(nca_1/(ntarg_1)*100))  

print("Number 2 of major vessels colored by flourosopy: {:.0f}%".format(nca_2/(ntarg_1)*100)) 

print("Number 3 of major vessels colored by flourosopy: {:.0f}%".format(nca_3/(ntarg_1)*100)) 

print()

line_labels = 'Number 0', 'Number 1', 'Number 2', 'Number 3'

sizes1 = [45, 44, 31, 18]



# Have disease by number of major vessels (0-3) colored by flourosopy

dtarg_1, dca_0, dca_1, dca_2, dca_3 = len(df[df.target == 1]), len(df[(df.target == 1) & (df.ca == 0)]), len(df[(df.target == 1) & (df.ca == 1)]), len(df[(df.target == 1) & (df.ca == 2)]), len(df[(df.target == 1) & (df.ca >= 3)])



print("Out of %s patients, that have heart disease have: %s - number 0 of major vessels colored by flourosopy,%s - number 1, %s - number 2, %s - number 3" %(dtarg_1, dca_0, dca_1, dca_2, dca_3))

print("Number 0 of major vessels colored by flourosopy: {:.0f}%".format(dca_0/(dtarg_1)*100)) 

print("Number 1 of major vessels colored by flourosopy: {:.0f}%".format(dca_1/(dtarg_1)*100))  

print("Number 2 of major vessels colored by flourosopy: {:.0f}%".format(dca_2/(dtarg_1)*100)) 

print("Number 3 of major vessels colored by flourosopy: {:.0f}%".format(dca_3/(dtarg_1)*100)) 



sizes2 = [130, 21, 7, 7]



fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 6))

fig.patch.set_facecolor('darkgrey')

l1 = ax1.pie(sizes1, autopct='%1.0f%%',

        shadow=True, startangle=90)

l2 = ax2.pie(sizes2, autopct='%1.0f%%',

        shadow=True, explode=(0, 0, 0, 0))

ax1.set_title('Dont have disease')

ax2.set_title('Have disease')

fig.legend([l1, l2],    

           labels=line_labels, 

           loc="center left",  

           borderaxespad=0.1,   

           title="Major vessels" 

           )



plt.subplots_adjust(right=0.9)

plt.show()

print("Patients, that have heart disease mostly have number 0 of major vessels colored by flourosopy")

print("Only 1 of 3 patients, that dont have heart disease have number 0 of major vessels colored by flourosopy ant they more likely to have number 1, 2 and 3 of major vessels colored by flourosopy")

print("                    Patients by thal       ") 

print()

# Dont have disease by thal: 3 = normal; 6 = fixed defect; 7 = reversable defect

ntarg_1, nthal_0, nthal_1, nthal_2, nthal_3 = len(df[df.target == 0]), len(df[(df.target == 0) & (df.thal == 0)]), len(df[(df.target == 0) & (df.thal == 1)]), len(df[(df.target == 0) & (df.thal == 2)]), len(df[(df.target == 0) & (df.thal == 3)])





print("Out of %s patients, that dont have heart disease have: %s - thal 0, %s - thal 1, %s - thal 2, %s - thal 3" %(ntarg_1, nthal_0, nthal_1, nthal_2, nthal_3))

print("thal 0: {:.0f}%".format(nthal_0/(ntarg_1)*100)) 

print("thal 1: {:.0f}%".format(nthal_1/(ntarg_1)*100))  

print("thal 2: {:.0f}%".format(nthal_2/(ntarg_1)*100)) 

print("thal 3: {:.0f}%".format(nthal_3/(ntarg_1)*100)) 

print()

line_labels = 'thal 0', 'thal 1', 'thal 2', 'thal 3'

sizes1 = [1, 12, 36, 89]



# Have disease by thal: 3 = normal; 6 = fixed defect; 7 = reversable defect

dtarg_1, dthal_0, dthal_1, dthal_2, dthal_3 = len(df[df.target == 1]), len(df[(df.target == 1) & (df.thal == 0)]), len(df[(df.target == 1) & (df.thal == 1)]), len(df[(df.target == 1) & (df.thal == 2)]), len(df[(df.target == 1) & (df.thal == 3)])





print("Out of %s patients, that have heart disease have: %s - thal 0, %s - thal 1, %s - thal 2, %s - thal 3" %(dtarg_1, dthal_0, dthal_1, dthal_2, dthal_3))

print("thal 0: {:.0f}%".format(dthal_0/(dtarg_1)*100)) 

print("thal 1: {:.0f}%".format(dthal_1/(dtarg_1)*100))  

print("thal 2: {:.0f}%".format(dthal_2/(dtarg_1)*100)) 

print("thal 3: {:.0f}%".format(dthal_3/(dtarg_1)*100)) 



sizes2 = [1, 6, 130, 28]



fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 6))

fig.patch.set_facecolor('darkgrey')

l1 = ax1.pie(sizes1, autopct='%1.0f%%',

        shadow=True, startangle=90)

l2 = ax2.pie(sizes2, autopct='%1.0f%%',

        shadow=True, explode=(0, 0, 0, 0))

ax1.set_title('Dont have disease')

ax2.set_title('Have disease')

fig.legend([l1, l2],    

           labels=line_labels, 

           loc="center left",  

           borderaxespad=0.1,   

           title="By thal" 

           )



plt.subplots_adjust(right=0.9)

plt.show()

print("Almost 4 of 5 patients, that have heart disease have thal 2")

print("Almost 2 of 3 patients, that dont have heart disease have thal 3")
X = df.drop('target', axis=1)

y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
X_train[:1]
classifiers = { 

        'Random Forest':RandomForestClassifier(n_estimators=10, random_state=0), 

        'SVM':SVC(gamma=0.01, kernel='linear'), 'MLP' : MLPClassifier(hidden_layer_sizes=(11,11,11), max_iter=500), 'Logistic Regression': LogisticRegression(solver='liblinear'), 

        'Decision Tree':DecisionTreeClassifier(), 'Gradient Boosting' :GradientBoostingClassifier(max_features=1, learning_rate=0.05) 

       }



predict_value = {}

for k, v in classifiers.items():

    model = v

    model.fit(X_train, y_train)

    predict_value[k] = model.score(X_test, y_test)*100

    print('Acurracy of ' + k + ' is {0:.2f}'.format(model.score(X_test, y_test)*100))

plt.style.use('ggplot')

plt.figure(figsize=(16, 6))

sns.barplot(x=list(predict_value.keys()), y=list(predict_value.values()), palette="rocket")

sns.despine(bottom=True)

plt.setp(f.axes, yticks=[])

plt.tight_layout(h_pad=2)



plt.yticks(np.arange(0,100,5))

plt.ylabel("Accuracy")

plt.xlabel("Models")

plt.show()
GBC = GradientBoostingClassifier(max_features=1, learning_rate=0.05)

GBC.fit(X_train, y_train)

GBC_pred = GBC.predict(X_test)

print(classification_report(y_test, GBC_pred))

print(confusion_matrix(y_test, GBC_pred))

print()
Xnew = [[34, 0, 0, 124, 160, 1, 0, 130, 0, 3.1, 0, 0, 1]]

Xnew = sc.transform(Xnew)

Ynew = GBC.predict(Xnew)

print(Ynew, "Diagnosis ({} for Non-disease, {} for disease)".format(0, 1))
