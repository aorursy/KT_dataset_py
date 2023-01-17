import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import numpy as np
%matplotlib inline

BlackFriday=pd.read_csv('../input/BlackFriday.csv')
BlackFriday.head()
fun={'Purchase':{'Purchase':'sum'}}
groupby=BlackFriday.groupby(["Age","Gender"]).agg(fun).reset_index()
groupby.columns=groupby.columns.droplevel(1)
groupby.count()
ind =np.arange(0,groupby.count().Gender,2)
width=0.5
fig, ax = plt.subplots()
rects1 = ax.bar(ind, groupby[groupby['Gender']=='F'].Purchase, width, color='r')
rects2 = ax.bar(ind + width, groupby[groupby['Gender']=='M'].Purchase, width, color='b')
ax.set_ylabel('Total Purchase')
ax.set_xlabel('Age Groups')

ax.set_title('Expenditure of People of Different Age and Sex on Black Friday\n')
ax.legend((rects1[0], rects2[0]), ('Female', 'Male'))
ax.set_xticks(ind +0.1)
ax.set_xticklabels(groupby['Age'])
plt.grid(True)
plt.show()

fun={'User_ID':{'Count':'count'}}
Age_Count=BlackFriday.groupby("Age").agg(fun).reset_index()
Age_Count.columns=Age_Count.columns.droplevel(1)
plt.plot(Age_Count.User_ID)
plt.scatter(Age_Count['Age'],Age_Count['User_ID'])
plt.title("Number of Users in each Age group")
plt.xlabel('Age Group')
plt.ylabel('Number of Users')
plt.grid(True)
plt.show()
BlackFriday.boxplot(column='Purchase', by=['Product_Category_1'])
plt.xticks(rotation=90)
BlackFriday.boxplot(column='Purchase', by=['Product_Category_2'])
plt.xticks(rotation=90)
BlackFriday.boxplot(column='Purchase', by=['Product_Category_3'])
plt.xticks(rotation=90)

fun={'Purchase':{'Purchase':'sum'},'User_ID':{'Users':'count'}}
group=BlackFriday.groupby('Product_ID').agg(fun).reset_index()
group.columns=group.columns.droplevel(0)
group.rename(columns = {'':'Product_ID'},inplace = True) 
print(group[group.Users==group.Users.max()])
print(group[group.Purchase==group.Purchase.max()])