import numpy as np

import scipy.stats as st

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

from sklearn import preprocessing

sns.set_style("whitegrid", {'axes.grid' : False})

from statsmodels.distributions.empirical_distribution import ECDF

import statsmodels.api as sm

import math

df = pd.read_csv("../input/video-game-sales-with-ratings/Video_Games_Sales_as_at_22_Dec_2016.csv") # Read the CSV from the data file
df.shape #Get the number of rows and columns
df.info() #Get the data types of the columns
columns_for_analysis = ["User_Score","User_Count", "Critic_Score", "Global_Sales"]



df['User_Score'] =  pd.to_numeric(df['User_Score'], errors='coerce')

def remove_zero_nan_values(df,list_of_columns): #

    df = df

    for i in list_of_columns:

        df = df[df[i].notnull()]

        df = df[df[i]>= 0]

    return df
def store_columns_in_array(df,list_of_columns):

    x = []

    for i in list_of_columns:

        y = df[i].values

        x.append(y)

    x = np.array(x).T

    return x
df = remove_zero_nan_values(df,columns_for_analysis)

X = store_columns_in_array(df,columns_for_analysis)
len(X)
user_reviews = X[:,0]
np.mean(user_reviews)
np.std(user_reviews)
ss = np.sqrt(np.var(user_reviews))

st.moment(user_reviews,3)/(ss**3)
st.moment(user_reviews,4)/(ss**4)
plt.figure(figsize=(6,4))

ax = sns.distplot(user_reviews,rug=True,

                 kde_kws={"label": "Kernel Density", "color" : 'k'},

                 hist_kws={"label": "Histogram", "color" : 'c'}

                )

plt.xlim([0,10])



sns.rugplot(np.array([1]),label="Rug Plot")

plt.xlabel('User Review Rating (x)')

plt.ylabel('Estimated Density')

xk,yk = ax.get_lines()[0].get_data()

mm = np.mean(user_reviews)

md = np.median(user_reviews)

mo = xk[np.argmax(yk)]

xx = np.ones(2)

yy = np.array([0, 0.5])

plt.xticks(np.arange(0, 11, step=1))

plt.plot(mm*xx,yy,'--b',label='Mean', alpha=0.8)

plt.plot(md*xx,yy,'-.r',label='Median', alpha=0.8)

plt.plot(mo*xx,yy,':m',label='KDE-estimated Mode', alpha=0.8)

plt.legend()

plt.tight_layout()

plt.savefig('kde.png',format='png')

plt.show()
plt.figure(figsize=(6,4))

ecdf=ECDF(user_reviews)



p25 = np.percentile(user_reviews,25)

p50 = np.percentile(user_reviews,50)

p75 = np.percentile(user_reviews,75)

p100 = np.percentile(user_reviews,100)

plt.step(ecdf.x,100*ecdf.y,linestyle='--',c=[0.5,0.5,0.5])

plt.plot([0,p25,p25],[25, 25, 0],'-k')

plt.plot([0,p50,p50],[50, 50, 0],'-k')

plt.plot([0,p75,p75],[75, 75, 0],'-k')

plt.xlabel('User Review Rating (x)')

plt.ylabel('ECDF x 100%')

plt.ylim([0,100])

plt.xlim([0,5])

plt.xticks((0,p25,p50,p75, p100),('0',p25,p50,p75,p100))

results = [str(i) for i in plt.yticks()[0]]

results2 = [25,50,75]

plt.yticks((list(plt.yticks()[0]))+results2,results+['$_{P25}$','$_{P50}$','$_{P75}$'])

plt.tight_layout()

plt.savefig('ecdf.png',format='png')

plt.show()
user_reviews = user_reviews

critic_reviews = X[:,2]
plt.figure(figsize=(6,4))

plt.scatter(user_reviews,critic_reviews, s=7)

plt.xlabel('User Reviews')

plt.ylabel('Critic Reviews')

plt.savefig('scatter.png',format='png')

plt.show()
sns.kdeplot(user_reviews,critic_reviews,cmap="Blues")

plt.xlabel('User Reviews')

plt.ylabel('Critic Reviews')

plt.savefig('kde_2.png',format='png')

plt.show()
np.corrcoef(np.array([user_reviews,critic_reviews]).T,rowvar=False) 
values_to_center = np.array([user_reviews,critic_reviews]).T
mean_to_subtract = np.mean(values_to_center, axis=0)
values_to_center = values_to_center - mean_to_subtract
np.mean(values_to_center, axis=0) #Check if the centered values equal 0


sns.kdeplot(values_to_center[:,0],values_to_center[:,1],cmap="Blues")

plt.xlabel('User Reviews')

plt.ylabel('Critic Reviews')

plt.savefig('kde_centered.png',format='png')

plt.show()


values_to_standardize = np.std(values_to_center, axis=0) 

final_values = values_to_center / values_to_standardize
sns.kdeplot(final_values[:,0],final_values[:,1],cmap="Blues")

plt.xlabel('User Reviews')

plt.ylabel('Critic Reviews')

plt.savefig('kde_standardised.png',format='png')

plt.show()
np.cov(final_values,rowvar=False) #Check if the matrix matches the Pearson product-moment correlation coefficients.
np.corrcoef(values_to_center,rowvar=False) #Check if the matrix matches the co-variance of the standarised values
pd.DataFrame(np.corrcoef(X.T), columns=columns_for_analysis, index=columns_for_analysis)

global_sales = X[:,3]



h = sns.jointplot(user_reviews,global_sales,cmap="Blues",kind="hex")

h.set_axis_labels('User Reviews', 'Global Sales (Millions)', fontsize=10)

plt.subplots_adjust(left=0.1, right=0.8, top=0.8, bottom=0.1) 

cbar_ax = h.fig.add_axes([.85, .25, .05, .4])

plt.colorbar(cax=cbar_ax)

plt.savefig('outliers.png',format='png')

plt.show()
global_sales = X[:,3]



h = sns.jointplot(user_reviews,np.log(global_sales),cmap="Blues",kind="hex")

ax = h.ax_joint

h.set_axis_labels('User Reviews', 'Global Sales (Millions - natural log)', fontsize=10)

plt.subplots_adjust(left=0.1, right=0.8, top=0.8, bottom=0.1) 

cbar_ax = h.fig.add_axes([.85, .25, .05, .4])



plt.colorbar(cax=cbar_ax)

plt.savefig('outliers_log.png',format='png')

plt.show()
