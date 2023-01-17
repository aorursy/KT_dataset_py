import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
%matplotlib inline
df = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
df.head()
class Proportion:
    def __init__(self,n_yes,n,z):
        """Z multiplier from appropriate distribution based on desired confidence level and sample design"""
        self.n_yes = n_yes
        self.n=n
        self.z=z
        self.best_estimate = round(n_yes/n,2)
        
    def estimated_standard_error(self):
        import numpy as np
        return np.sqrt((self.best_estimate*(1-self.best_estimate))/self.n)
    
    def margin_of_error(self):
        return self.z*self.estimated_standard_error()

    def proportion(self):
        lcb = self.best_estimate - (self.z*self.estimated_standard_error())
        ucb = self.best_estimate + (self.z*self.estimated_standard_error())
        return (lcb,ucb)
class Mean:
    import numpy as np
    def __init__(self,mean,std,n,t):
        '''t multiplier comes from t distribution with n-1 degree of freedom'''
        self.best_estimate = mean
        self.std = std
        self.n = n
        self.t = t
        self.estimated_se = self.std / np.sqrt(self.n)
        
    def moe(self):
        return self.t * self.estimated_se
    
    def mean(self):
        lcb = self.best_estimate - self.moe()
        ucb = self.best_estimate + self.moe()
        return (lcb,ucb)
class diffMean:
    import numpy as np
    def __init__(self,x1,x2,std1,std2,n1,n2,t):
        '''t multiplier comes from t distribution with appropriate degree of freedom'''        
        self.best_estimate1 = x1
        self.best_estimate2 = x2
        self.std1 = std1
        self.std2 = std2
        self.n1 = n1
        self.n2 = n2
        self.t = t
        self.pooled_estimated_se = np.sqrt((np.sqrt(((self.n1-1)*(self.std1**2) + (self.n2-1)*(self.std2**2)) / ((self.n1+self.n2)-2))) * (np.sqrt((1/self.n1)+(1/self.n2))))
        self.unpooled_estimated_se = np.sqrt(((self.std1**2)/self.n1) + ((self.std2**2)/self.n2))
        
    def pooledMoe(self):
        return (self.t)*(self.pooled_estimated_se)
    
    def pooledMean(self):
        lcb = (self.best_estimate1 - self.best_estimate2) - self.pooledMoe()
        ucb = (self.best_estimate1 - self.best_estimate2) + self.pooledMoe()
        return (lcb,ucb)
    
    def unpooledMoe(self):
        return self.t * self.unpooled_estimated_se
    
    def unpooledMean(self):
        lcb = (self.best_estimate1 - self.best_estimate2) - self.unpooledMoe()
        ucb = (self.best_estimate1 - self.best_estimate2) + self.unpooledMoe()
        return (lcb,ucb)
df.Outcome.replace({0:'Non-Diab',1:'Diab'},inplace=True)
print(df.Outcome.value_counts())
n = df.shape[0]
diabetic = df.Outcome.value_counts().loc['Diab']

print("\nTotal Observation ==>",n,"\t","Number of Diabetic Patient==> ",diabetic,"\n")

diab_pro = Proportion(diabetic, n ,1.96)  # I am using z=1.96 for 95% C.I.
print("\nBest Point Estimate for Proportion of People with Diabetes==>", diab_pro.best_estimate*100)
print("\nEstimated Standard Error for Proportion of People with Diabetes==>",diab_pro.estimated_standard_error())
print("\nMargin of Error is for Proportion of People with Diabetes ==>",diab_pro.margin_of_error())
print("\n95% Confidence Interval for Proportion of People with Diabetes ==> ",diab_pro.proportion(),"\n")
import statsmodels.api as sm
print("\n95% Confidence interval with statsmodels library ==>",sm.stats.proportion_confint(diabetic, n),"\n")
df_diabetic = df[df.Outcome=='Diab']
df_diabetic.head()
mean_preg_diab = Mean(df_diabetic['Pregnancies'].mean(),df_diabetic['Pregnancies'].std(),df_diabetic.shape[0],1.962)
print("\nBest point estimate for Mean Pregnancy Month of Patients with diabetes ==>", mean_preg_diab.best_estimate)
print("\nEstimated Standard Error for Mean Pregnancy Month of Patients with diabetes ==>",mean_preg_diab.estimated_se)
print("\nMargin of Error for Mean Pregnancy Month of Patients with diabetes ==>",mean_preg_diab.moe())
print("\n95% Confidence Interval for Mean Pregnancy Month of Patients with diabetes ==> ",mean_preg_diab.mean(),"\n")
print("\n95% C.I. with statsmodels library ==>",sm.stats.DescrStatsW(df_diabetic['Pregnancies']).zconfint_mean())
plt.figure(dpi=120,figsize=(5,3))
sns.distplot(df_diabetic['Pregnancies'],color='green')
plt.axvline(x=4.417756079185482,color = 'black',ls=':')
plt.axvline(x=5.313587204396608,color = 'black',ls=':')
plt.axvline(x=mean_preg_diab.best_estimate,color='red',ls='--')
plt.xticks([4.417756079185482,5.313587204396608],['lcb','ucb'],rotation=90)
plt.xlabel('Pregnancy Class for Diabetic Patients',fontdict={'fontsize':8})
plt.ylabel('Count/Distribution',fontdict={'fontsize':8})
plt.title('Pregnancies Distribution for Diabetic Patients',fontdict={'fontsize':8}) 
plt.show()
df_non_diabetic = df[df.Outcome=='Non-Diab']
df_non_diabetic.head()
mean_preg_non_diab = Mean(df_non_diabetic['Pregnancies'].mean(),df_non_diabetic['Pregnancies'].std(),df_non_diabetic.shape[0],1.962)
print("\nBest point estimate for Mean Pregnancy Month of non diabetic patients ==>", mean_preg_non_diab.best_estimate)
print("\nEstimated Standard Error for Mean Pregnancy Month of non diabetic patients ==>",mean_preg_non_diab.estimated_se)
print("\nMargin of Error for Mean Pregnancy Month of non diabetic patients ==>",mean_preg_non_diab.moe())
print("\n95% Confidence Interval for Mean Pregnancy Month of non diabetic patients ==> ",mean_preg_non_diab.mean(),"\n")
print("\n95% C.I. with statsmodels library ==>",sm.stats.DescrStatsW(df_non_diabetic['Pregnancies']).zconfint_mean())
plt.figure(dpi=120,figsize=(5,3))
sns.distplot(df_non_diabetic['Pregnancies'],color='green')
plt.axvline(x=3.0332622455725544,color = 'black',ls=':')
plt.axvline(x=3.5627377544274457,color = 'black',ls=':')
plt.axvline(x=mean_preg_non_diab.best_estimate,color='red',ls='--')
plt.xticks([3.0332622455725544, 3.5627377544274457],['lcb','ucb'],rotation=90)
plt.xlabel('Pregnancy Class for Non Diabetic Patients',fontdict={'fontsize':8})
plt.ylabel('Count/Distribution',fontdict={'fontsize':8})
plt.title('Pregnancies Distribution for Non Diabetic Patients',fontdict={'fontsize':8}) 
plt.show()
print("\nSample Information==>")
df.groupby('Outcome').describe()['Pregnancies'].transpose().loc[['mean','std'],:]
fig,axes = plt.subplots(nrows=1,ncols=2,dpi=100,figsize = (9,5))

plot0 = sns.boxplot(df_diabetic['Pregnancies'],ax=axes[0],orient='v',color = 'red')
axes[0].set_title('Pregnancies',fontdict={'fontsize':8})
axes[0].set_xlabel('Diabetic',fontdict={'fontsize':7})
axes[0].set_ylabel('Five Point Summary',fontdict={'fontsize':7})
plt.tight_layout()

plot1 = sns.boxplot(df_non_diabetic['Pregnancies'],ax=axes[1],orient='v',color='green')
axes[1].set_title('Pregnancies',fontdict={'fontsize':8})
axes[1].set_xlabel('Non Diabetic',fontdict={'fontsize':7})
axes[1].set_ylabel('Five Point Summary',fontdict={'fontsize':7})
plt.tight_layout()

x1 = df_diabetic.Pregnancies.mean()
x2 = df_non_diabetic.Pregnancies.mean()
std1 = df_diabetic.Pregnancies.std()
std2 = df_non_diabetic.Pregnancies.std()
n1 = df_diabetic.shape[0]
n2 = df_non_diabetic.shape[0]
mean_diff_preg = diffMean(x1,x2,std1,std2,n1,n2,1.98)
mean_diff_preg.pooledMean()


print("\nBest point estimate for (μ1 − μ2 ): Pregnancies Month ==>", (mean_diff_preg.best_estimate1-mean_diff_preg.best_estimate2))
print("\nEstimated Standard Error for (μ1 − μ2 ): Pregnancies Month ==>",mean_diff_preg.pooled_estimated_se)
print("\nMargin of Error for (μ1 − μ2 ): Pregnancies Month ==>",mean_diff_preg.pooledMoe())
print("\n95% Confidence Interval for (μ1 − μ2 ): Pregnancies Month ==> ",mean_diff_preg.pooledMean(),"\n")
import statsmodels.api as sm
z,p_value = sm.stats.ztest(df_diabetic['Pregnancies'],df_non_diabetic['Pregnancies'])

print("P-Value is ==> ",p_value)