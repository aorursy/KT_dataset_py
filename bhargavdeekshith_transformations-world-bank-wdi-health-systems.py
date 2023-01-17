import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np



df = pd.read_csv("../input/world-bank-wdi-212-health-systems/2.12_Health_systems.csv")

df.head()
df.isnull().sum()
#median used for imputation
def imputer(column):
    return df[column].fillna(value = df[column].median(),inplace=True)
df.columns
numerical_cols = [['Health_exp_pct_GDP_2016', 'Health_exp_public_pct_2016',
       'Health_exp_out_of_pocket_pct_2016', 'Health_exp_per_capita_USD_2016',
       'per_capita_exp_PPP_2016', 'External_health_exp_pct_2016',
       'Physicians_per_1000_2009-18', 'Nurse_midwife_per_1000_2009-18',
       'Specialist_surgical_per_1000_2008-18',
       'Completeness_of_birth_reg_2009-18',
       'Completeness_of_death_reg_2008-16']]
df.isnull().sum()
imputer('Health_exp_pct_GDP_2016')
imputer('Health_exp_public_pct_2016')
imputer('Health_exp_public_pct_2016')
imputer('Health_exp_out_of_pocket_pct_2016')
imputer('Health_exp_per_capita_USD_2016')
imputer('per_capita_exp_PPP_2016')
imputer('External_health_exp_pct_2016')
imputer('Physicians_per_1000_2009-18')
imputer('Nurse_midwife_per_1000_2009-18')
imputer('Specialist_surgical_per_1000_2008-18')
imputer('Completeness_of_birth_reg_2009-18')
imputer('Completeness_of_death_reg_2008-16')
df.isnull().sum()
df.skew()
import scipy.stats as stats
def transformations(column):
    plt.figure(figsize=(22,25))
    plt.tight_layout
    plt.subplot(8,2,1)
    plt.hist(df[column])
    plt.title('Original distribution')
    plt.subplot(8,2,2)
    stats.probplot(df[column],dist='norm',plot=plt)
    raw_skewness = df[column].skew()
    
    log_transform = np.log(df[column]+1)
    plt.subplot(8,2,3)
    plt.hist(log_transform)
    plt.title('Log transformation')
    plt.subplot(8,2,4)
    stats.probplot(log_transform,dist='norm',plot=plt)
    
    recip_transform = 1/(df[column]+1)
    plt.subplot(8,2,5)
    plt.hist(recip_transform)
    plt.title('Reciprocal transformation')
    plt.subplot(8,2,6)
    stats.probplot(recip_transform,dist='norm',plot=plt)
    
    exp_2 = df[column]**0.2
    exp_3 = df[column]**0.3
    plt.subplot(8,2,7)
    plt.hist(exp_2)
    plt.title('exp_2 transformation')
    plt.subplot(8,2,8)
    stats.probplot(exp_2,dist='norm',plot=plt)
    plt.subplot(8,2,9)
    plt.hist(exp_3)
    plt.title('exp_3 transformation')
    plt.subplot(8,2,10)
    stats.probplot(exp_3,dist='norm',plot=plt)
    
    sqrt_transform = df[column]**(1/2)
    cube_transform = df[column]**(1/3)
    plt.subplot(8,2,11)
    plt.hist(sqrt_transform)
    plt.title('square root transformation')
    plt.subplot(8,2,12)
    stats.probplot(sqrt_transform,dist='norm',plot=plt)
    plt.subplot(8,2,13)
    plt.hist(cube_transform)
    plt.title('cube root transformation')
    plt.subplot(8,2,14)
    stats.probplot(cube_transform,dist='norm',plot=plt)
    
    df[column],param = stats.boxcox(df[column]+1)
    boxcox_skewness = df[column].skew()
    plt.subplot(8,2,15)
    plt.hist(df[column])
    plt.title('Boxcox Transformation')
    plt.subplot(8,2,16)
    stats.probplot(df[column],dist='norm',plot=plt)
    
    print(pd.DataFrame({'Method':['No transformation','Log','Reciprocal','EXP_0.2','EXP_0.3','Square Root','Cube Root','Boxcox'],
                        'Skewness':[raw_skewness,log_transform.skew(),recip_transform.skew(),exp_2.skew(),exp_3.skew(),sqrt_transform.skew(),cube_transform.skew(),boxcox_skewness]}))
transformations('Health_exp_pct_GDP_2016')
transformations('Health_exp_public_pct_2016')
transformations('Health_exp_out_of_pocket_pct_2016')
transformations('Health_exp_per_capita_USD_2016')
transformations('per_capita_exp_PPP_2016')
transformations('External_health_exp_pct_2016')
transformations('Physicians_per_1000_2009-18')
transformations('Nurse_midwife_per_1000_2009-18')
transformations('Specialist_surgical_per_1000_2008-18')
transformations('Completeness_of_birth_reg_2009-18')
transformations('Completeness_of_death_reg_2008-16')