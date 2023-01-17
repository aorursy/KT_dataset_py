import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')
pdf = pd.read_csv("../input/HR_comma_sep.csv")

pdf.head(10)
pdf.isnull().any()
pdf.salary = pdf.salary.map( {'medium': 1, 'low': 0, 'high': 2} ).astype(int)

pdf.sales = pdf.sales.map({'IT': 0, 'support': 1, 'marketing': 2, 'technical': 3, 'management': 4, 'hr': 5, 'RandD': 5, 'product_mng': 6, 'sales': 7, 'accounting': 8})
print(set(pdf.sales))
pdf.describe()
colormap = plt.cm.viridis

plt.figure(figsize=(8,8))

plt.title('Pearson Correlation of Features', y=1.05, size=20)

sns.heatmap(pdf.astype(float).corr(), linewidths=1, vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)

plt.show()
left = pdf[pdf.left == 1]

print(len(left))
left.head(10)
pdf.columns
plt.scatter(left.average_montly_hours, left.last_evaluation, marker='.', color='b')

plt.xlabel("Average Monthly Hours")

plt.ylabel("Last Evaluation")

plt.title("Scatter plot for Employees who left the firm")

plt.show()
bucket_le = []

bucket_sl = []

num_proj = [i for i in range(min(left.number_project), max(left.number_project)+1)]

for index in range(min(left.number_project), max(left.number_project)+1):

    bucket_le.append(np.mean(left.last_evaluation[left.number_project == index]))

    bucket_sl.append(np.mean(left.satisfaction_level[left.number_project == index]))

plt.scatter(num_proj, bucket_le, marker='.', color='b', s=200)

plt.plot(num_proj, bucket_le, color='b',  linewidth=1)

plt.scatter(num_proj, bucket_sl, marker='.', color='r', s=200)

plt.plot(num_proj, bucket_sl, color='r',  linewidth=1)

plt.xlabel("Number of project")

plt.ylabel("Last Evaluation")

plt.title("Line Graph for Employees who left the firm")

plt.legend(["Last Evaluation", "Satisfaction Level"], loc=3)

axes = plt.gca()

axes.set_ylim([0, 1])

plt.show()
plt.scatter(left.satisfaction_level, left.last_evaluation, marker='o', color='r')

plt.xlabel("Satisfaction level")

plt.ylabel("Last Evaluation")

plt.title("Scatter plot for Employees whos left the firm")

plt.show()
kmeans_df = left.copy()



del kmeans_df["number_project"]

del kmeans_df["average_montly_hours"]

del kmeans_df["time_spend_company"]

del kmeans_df["Work_accident"]

del kmeans_df["left"]

del kmeans_df["promotion_last_5years"]

del kmeans_df["sales"]

del kmeans_df["salary"]

              

kmeans_df.head()
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 3, random_state = 0).fit(kmeans_df)

print(kmeans.cluster_centers_)



left['label'] = kmeans.labels_
plt.figure()

plt.xlabel('Satisfaction Level')

plt.ylabel('Last Evaluation')

plt.title('3 Clusters of employees who left')

plt.scatter(x=kmeans.cluster_centers_[:,0],y=kmeans.cluster_centers_[:,1],color="black",marker="X",s=200)

plt.plot(left.satisfaction_level[left.label==0],left.last_evaluation[left.label==0],'o', alpha = 0.15, color = 'r')

plt.plot(left.satisfaction_level[left.label==1],left.last_evaluation[left.label==1],'o', alpha = 0.07, color = 'y')

plt.plot(left.satisfaction_level[left.label==2],left.last_evaluation[left.label==2],'o', alpha = 0.08, color = 'b')

plt.legend(['Winners','Frustrated','Bad Match'], loc = 4, fontsize = 15,frameon=True)

plt.show()
winners    = left[left.label == 0]

frustrated = left[left.label == 1]

bad_match  = left[left.label == 2]
# function to get number of employees in each department

def employee_count(emp):

    emp_cnt = []

    for index in range(len(np.unique(emp.sales))):

        num = len(emp[emp.sales == index])

        emp_cnt.append(num)

    return emp_cnt
winners_count       = employee_count(winners)

frustrated_count    = employee_count(frustrated)

bad_match_count     = employee_count(bad_match)



departments = ['IT','support','marketing','technical','management','hr','RandD','product_mng','sales']

values = np.unique(left.sales)

plt.bar(range(len(values)), winners_count, width = 0.25, color = 'r')

plt.bar(range(len(values))+0.25*np.ones(len(values)),frustrated_count, width = 0.25, color = 'y')

plt.bar(range(len(values))+0.5*np.ones(len(values)), bad_match_count, width = 0.25, color = 'b')



plt.xticks(range(len(values))+ 0.5*np.ones(len(values)), departments, rotation= 45)

plt.legend(['Winners','Frustrated','Bad Match'], loc = 2)



plt.title('Employees count from Different Departments')

plt.show()
def get_average_daily_working_hours(emp_df):

    avg_working_hours = []

    for index in range(len(np.unique(emp_df.sales))):

        hours = int(int(np.mean(emp_df.average_montly_hours[emp_df.sales == index]))/30)

        avg_working_hours.append(hours)

    return avg_working_hours
winners_avg_working_hours      = get_average_daily_working_hours(winners)

frustrated_avg_working_hours   = get_average_daily_working_hours(frustrated)

bad_match_avg_working_hours    = get_average_daily_working_hours(bad_match)



plt.figure(1)



plt.subplot(121)

departments = ['IT','support','marketing','technical','management','hr','RandD','product_mng','sales']

values = np.unique(left.sales)

plt.bar(range(len(values)), winners_avg_working_hours, width = 0.25, color = 'r')

plt.bar(range(len(values))+0.25*np.ones(len(values)),frustrated_avg_working_hours, width = 0.25, color = 'y')

plt.bar(range(len(values))+0.5*np.ones(len(values)), bad_match_avg_working_hours, width = 0.25, color = 'b')

plt.xticks(range(len(values))+ 0.5*np.ones(len(values)), departments, rotation= 45)

plt.legend(['Winners','Frustrated','Bad Match'], loc = 2)

axes = plt.gca()

axes.set_ylim([0, 11])

plt.title('Average Daily Hours by Employees from Different Departments')

plt.grid(True)



plt.subplot(122)

plt.bar(range(1), np.mean(winners_avg_working_hours), width = 0.3, color = 'r')

plt.bar(range(1) + 0.4*np.ones(1), np.mean(frustrated_avg_working_hours), width=0.3, color='y')

plt.bar(range(1) + 0.8*np.ones(1), np.mean(bad_match_avg_working_hours), width=0.3, color='b')

plt.xticks([0, 0.4, 0.8], ["winners", "frustrated", "bad_match"], size = 10)

plt.xlabel("Groups")

plt.ylabel("Average working hours daily")

plt.title("Plot to show average working hours daily of the groups")



plt.subplots_adjust(top=0.98, bottom=0.08, left=0.05, right=2, hspace=0.15,wspace=0.15)

plt.show()
plt.figure()

import seaborn as sns

sns.kdeplot(winners.average_montly_hours, color = 'b')

sns.kdeplot(frustrated.average_montly_hours, color ='r')

sns.kdeplot(bad_match.average_montly_hours, color ='y')

plt.legend(['Winners', 'Frustrated', 'Bad Match'])

plt.title('Average Monthly Hours distribution')

plt.show()
# function to get number of employees in each department

def get_average_time_spent_in_company(emp):

    time_cnt = []

    for index in range(len(np.unique(emp.sales))):

        num = np.mean(emp.time_spend_company[emp.sales == index])

        time_cnt.append(num)

    return time_cnt
winners_years_count       = get_average_time_spent_in_company(winners)

frustrated_years_count    = get_average_time_spent_in_company(frustrated)

bad_match_years_count     = get_average_time_spent_in_company(bad_match)



plt.figure(1)



plt.subplot(121)

departments = ['IT','support','marketing','technical','management','hr','RandD','product_mng','sales']

values = np.unique(left.sales)

plt.bar(range(len(values)), winners_years_count, width = 0.25, color = 'r')

plt.bar(range(len(values))+0.25*np.ones(len(values)),frustrated_years_count, width = 0.25, color = 'y')

plt.bar(range(len(values))+0.5*np.ones(len(values)), bad_match_years_count, width = 0.25, color = 'b')



plt.xticks(range(len(values))+ 0.5*np.ones(len(values)), departments, rotation= 45)

plt.legend(['Winners','Frustrated','Bad Match'], loc = 2)

plt.xlabel("Department")

plt.ylabel("Number of years")

plt.title('Average Time spent in company')

axes = plt.gca()

axes.set_ylim([0,7])

plt.grid(True)



plt.subplot(122)

plt.bar(range(1), np.mean(winners_years_count), width = 0.3, color = 'r')

plt.bar(range(1) + 0.4*np.ones(1), np.mean(frustrated_years_count), width=0.3, color='y')

plt.bar(range(1) + 0.8*np.ones(1), np.mean(bad_match_years_count), width=0.3, color='b')

plt.xticks([0, 0.4, 0.8], ["winners", "frustrated", "bad_match"], size = 10)

plt.xlabel("Groups")

plt.ylabel("Average Time spent in Company(Years)")

plt.title("Plot to show Average Time spent in company")

plt.tight_layout()



plt.subplots_adjust(top=0.98, bottom=0.08, left=0.05, right=2, hspace=0.15,wspace=0.15)

plt.show()
# function to get number of employees in each department

def get_average_number_of_projects(emp):

    project_cnt = []

    for index in range(len(np.unique(emp.sales))):

        num = np.mean(emp.Work_accident[emp.sales == index])

        project_cnt.append(num)

    return project_cnt
winners_project_count       = get_average_time_spent_in_company(winners)

frustrated_project_count    = get_average_time_spent_in_company(frustrated)

bad_match_project_count     = get_average_time_spent_in_company(bad_match)

plt.figure(1)

plt.subplot(121)

departments = ['IT','support','marketing','technical','management','hr','RandD','product_mng','sales']

values = np.unique(left.sales)

plt.bar(range(len(values)), winners_project_count, width = 0.25, color = 'r')

plt.bar(range(len(values))+0.25*np.ones(len(values)),frustrated_project_count, width = 0.25, color = 'y')

plt.bar(range(len(values))+0.5*np.ones(len(values)), bad_match_project_count, width = 0.25, color = 'b')



plt.xticks(range(len(values))+ 0.5*np.ones(len(values)), departments, rotation= 45)

plt.legend(['Winners','Frustrated','Bad Match'], loc = 2)

plt.xlabel("Department")

plt.ylabel("Number of Projects")

plt.title('Average Number of Projects done by employees')

axes = plt.gca()

axes.set_ylim([0,7])

plt.grid(True)



plt.subplot(122)

plt.bar(range(1), np.mean(winners_project_count), width = 0.3, color = 'r')

plt.bar(range(1) + 0.4*np.ones(1), np.mean(frustrated_project_count), width=0.3, color='y')

plt.bar(range(1) + 0.8*np.ones(1), np.mean(bad_match_project_count), width=0.3, color='b')

plt.xticks([0, 0.4, 0.8], ["winners", "frustrated", "bad_match"], size = 10)

plt.xlabel("Groups")

plt.ylabel("Average Time spent in Company(Years)")

plt.title("Plot to show Average Time spent in company")

plt.tight_layout()

plt.grid(True)



plt.subplots_adjust(top=0.98, bottom=0.08, left=0.05, right=2, hspace=0.15,wspace=0.15)

plt.show()
# function to get number of employees in each department

def get_average_promotions(emp):

    project_cnt = []

    for index in range(len(np.unique(emp.sales))):

        num = sum(emp.promotion_last_5years[emp.sales == index])

        project_cnt.append(num)

    return project_cnt
department_wise_promotions = get_average_promotions(left)

departments = ['IT','support','marketing','technical','management','hr','RandD','product_mng','sales']

values = np.unique(left.sales)

plt.subplot(221)

plt.plot(range(len(values)), department_wise_promotions, marker="X", markersize=15, color='r')

plt.xticks(range(len(values)), departments)

plt.legend(['Promotions'], loc = 2)

plt.xlabel("Department")

plt.ylabel("Number of Promotions")

axes = plt.gca()

axes.set_ylim([0,9])

plt.grid(True)



winners_promotions = get_average_promotions(winners)

frustrated_promotions = get_average_promotions(frustrated)

bad_match_promotions = get_average_promotions(bad_match)



plt.subplot(222)

plt.plot(range(len(values)), winners_promotions, marker="X", markersize=15, color='r')

plt.xticks(range(len(values)), departments)

plt.legend(['Winner Promotions'], loc = 2)

plt.xlabel("Department")

plt.ylabel("Number of Promotions")

axes = plt.gca()

axes.set_ylim([0,9])

plt.grid(True)



plt.subplot(223)

plt.plot(range(len(values)), frustrated_promotions, marker="X", markersize=15, color='y')

plt.xticks(range(len(values)), departments)

plt.legend(['Frustrated Promotions'], loc = 2)

plt.xlabel("Department")

plt.ylabel("Number of Promotions")

axes = plt.gca()

axes.set_ylim([0,9])

plt.grid(True)



plt.subplot(224)

plt.plot(range(len(values)), bad_match_promotions, marker="X", markersize=15, color='b')

plt.xticks(range(len(values)), departments)

plt.legend(['bad match Promotions'], loc = 2)

plt.xlabel("Department")

plt.ylabel("Number of Promotions")

axes = plt.gca()

axes.set_ylim([0,9])

plt.grid(True)



plt.subplots_adjust(top=2, bottom=0.3, left=0.05, right=2, hspace=0.15,wspace=0.15)

plt.show()
# function to get number of employees in each department

def get_average_work_accidents(emp):

    project_cnt = []

    for index in range(len(np.unique(emp.sales))):

        num = sum(emp.Work_accident[emp.sales == index])

        project_cnt.append(num)

    return project_cnt
department_wise_work_accident = get_average_work_accidents(left)

departments = ['IT','support','marketing','technical','management','hr','RandD','product_mng','sales']

values = np.unique(left.sales)

plt.subplot(221)

plt.plot(range(len(values)), department_wise_work_accident, marker="X", markersize=15, color='r')

plt.xticks(range(len(values)), departments)

plt.legend(['Promotions'], loc = 2)

plt.xlabel("Department")

plt.ylabel("Number of Work accidents")

axes = plt.gca()

axes.set_ylim([0,max(department_wise_work_accident)+2])

plt.grid(True)



winners_work_accident = get_average_work_accidents(winners)

frustrated_work_accident = get_average_work_accidents(frustrated)

bad_match_work_accident = get_average_work_accidents(bad_match)



plt.subplot(222)

plt.plot(range(len(values)), winners_work_accident, marker="X", markersize=15, color='r')

plt.xticks(range(len(values)), departments)

plt.legend(['Winner Promotions'], loc = 2)

plt.xlabel("Department")

plt.ylabel("Number of Work accidents")

axes = plt.gca()

axes.set_ylim([0,max(winners_work_accident)+2])

plt.grid(True)



plt.subplot(223)

plt.plot(range(len(values)), frustrated_work_accident, marker="X", markersize=15, color='y')

plt.xticks(range(len(values)), departments)

plt.legend(['Frustrated Promotions'], loc = 2)

plt.xlabel("Department")

plt.ylabel("Number of Work accidents")

axes = plt.gca()

axes.set_ylim([0,max(frustrated_work_accident)+2])

plt.grid(True)



plt.subplot(224)

plt.plot(range(len(values)), bad_match_work_accident, marker="X", markersize=15, color='b')

plt.xticks(range(len(values)), departments)

plt.legend(['bad match Promotions'], loc = 2)

plt.xlabel("Department")

plt.ylabel("Number of Work accidents")

axes = plt.gca()

axes.set_ylim([0,max(bad_match_work_accident)+2])

plt.grid(True)



plt.subplots_adjust(top=2, bottom=0.3, left=0.05, right=2, hspace=0.15,wspace=0.15)

plt.show()
plt.pie(department_wise_work_accident,

		labels = departments,

		shadow = True,		# gives shadow to the pie chart

		startangle = 0,		# sets the starting angle to 0

		explode = (0.1,0.1,0.1,0.1,0.1,0.1, 0.1, 0.1, 0.1),	# sets the peices out of the plot

		autopct = '%1.1f%%')	# automatically sets the percentage and displays it



plt.title('Work Accidents pie plot Graph')

plt.show()

def get_salary_stats(emp):

    sal_stat = []

    for index in range(np.unique(emp.sales)):

        num = sum(emp.Work_accident[emp.sales == index])

        project_cnt.append(num)

    return project_cnt
from collections import Counter

Winners_sal    = list(Counter(left.salary[left.label == 0]).values())

frustrated_sal = list(Counter(left.salary[left.label == 1]).values())

bad_match_sal  = list(Counter(left.salary[left.label == 2]).values())



print("Winners : " + str(Counter(left.salary[left.label == 0])))

print("Frustrated : " + str(Counter(left.salary[left.label == 1])))

print("Bad Match : " + str(Counter(left.salary[left.label == 2])))

print(Winners_sal)



plt.figure(1)

groups = ["WINNERS", "FRUSTRATED", "BAD MATCH"]



plt.subplot(131)

plt.ylabel('Number of Employees')

plt.bar( np.arange(3), [Winners_sal[0], frustrated_sal[0], bad_match_sal[0]], width = 0.25, color = 'b')

plt.xticks(range(3), groups,  rotation= 45)

plt.xlabel("Low Salary")



plt.subplot(132)

plt.title('Plot Showing Number of people in each salary group')

plt.bar( np.arange(3), [Winners_sal[1], frustrated_sal[1], bad_match_sal[1]], width = 0.25, color = 'y')

plt.xticks(range(3), groups,  rotation= 45)

plt.xlabel("Medium Salary")



plt.subplot(133)

plt.bar( np.arange(3), [Winners_sal[2], frustrated_sal[2], bad_match_sal[2]], width = 0.25, color = 'r')

plt.xticks(range(3), groups,  rotation= 45)

plt.subplots_adjust(left=0.05, right=2, hspace=0.15,wspace=0.15)

plt.xlabel("High Salary")

plt.show()
plt.figure()

import seaborn as sns

sns.kdeplot(winners.last_evaluation, color = 'b')

sns.kdeplot(frustrated.last_evaluation, color ='r')

sns.kdeplot(bad_match.last_evaluation, color ='y')

plt.legend(['Winners', 'Frustrated', 'Bad Match'])

plt.title('Last Evaluation distribution')

plt.show()
plt.figure()

import seaborn as sns

sns.kdeplot(winners.satisfaction_level, color = 'b')

sns.kdeplot(frustrated.satisfaction_level, color ='r')

sns.kdeplot(bad_match.satisfaction_level, color ='y')

plt.legend(['Winners', 'Frustrated', 'Bad Match'])

plt.title('Satisfaction_Level distribution')

plt.show()
# satisfaction_level	last_evaluation	number_project	average_montly_hours	time_spend_company	Work_accident	left	promotion_last_5years	sales	salary

plt.figure()

import seaborn as sns

sns.kdeplot(winners.number_project, color = 'b')

sns.kdeplot(frustrated.number_project, color ='r')

sns.kdeplot(bad_match.number_project, color ='y')

plt.legend(['Winners', 'Frustrated', 'Bad Match'])

plt.title('Number of Projects distribution')

plt.show()
# satisfaction_level	last_evaluation	number_project	average_montly_hours	time_spend_company	Work_accident	left	promotion_last_5years	sales	salary

plt.figure()

import seaborn as sns

sns.kdeplot(winners.time_spend_company, color = 'b')

sns.kdeplot(frustrated.time_spend_company, color ='r')

sns.kdeplot(bad_match.time_spend_company, color ='y')

plt.legend(['Winners', 'Frustrated', 'Bad Match'])

plt.title('Time Spent at company distribution')

plt.show()
from sklearn.preprocessing import StandardScaler, Normalizer, RobustScaler

from sklearn.decomposition import PCA



N_ss = StandardScaler()

N_ss.fit(pdf)

pdf_norm = N_ss.transform(pdf)
pca = PCA(n_components=2)

pca_representation = pca.fit_transform(pdf_norm)
left_colors = pdf["left"].map(lambda s : "g" if s==0 else "r")

plt.scatter(pca_representation[:,0],pca_representation[:,1],c = left_colors,alpha=0.5,s=20)

plt.title("Dimensionality reduction with PCA")

plt.legend(["Employee Left", "Employee Stayed"])

plt.show()
from sklearn.model_selection import cross_val_score, KFold

kf = KFold(n_splits=5, random_state=1, shuffle=True)

kf.get_n_splits(pdf.shape[0])
cols = pdf.columns

cols = cols.drop("left")

features = pdf[cols]

target = pdf["left"]
from sklearn.linear_model import LogisticRegression

clfModel1 = LogisticRegression(class_weight='balanced')

# check the accuracy

scores = cross_val_score(clfModel1,  features, target, cv=kf)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

print(scores)
from sklearn import svm

clfModel2 = svm.SVC()

# check the accuracy

scores = cross_val_score(clfModel2,  features, target, cv=kf)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

print(scores)
from sklearn import neighbors

n_neighbors = 3

clfModel3 = neighbors.KNeighborsClassifier(n_neighbors)

# the data is split 5 times

scores = cross_val_score(clfModel3,  features, target, cv=kf)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

print(scores)
from sklearn.ensemble import RandomForestClassifier

clfModel4 = RandomForestClassifier(n_estimators=14, max_depth=None, min_samples_split=2, random_state=0)

scores = cross_val_score(clfModel4,  features, target, cv=kf)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

print(scores)
from sklearn.ensemble import ExtraTreesClassifier

clfModel5 = ExtraTreesClassifier(n_estimators=15, max_depth=None, min_samples_split=2, random_state=0)

scores = cross_val_score(clfModel5,  features, target, cv=kf)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

print(scores)
from sklearn.ensemble import AdaBoostClassifier

clfModel6 = AdaBoostClassifier(n_estimators=100)

scores = cross_val_score(clfModel6,  features, target, cv=kf)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

print(scores)
from sklearn.tree import DecisionTreeClassifier

clfModel7 = DecisionTreeClassifier(max_depth=3)

scores = cross_val_score(clfModel6,  features, target, cv=kf, scoring="accuracy")

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

print(scores)