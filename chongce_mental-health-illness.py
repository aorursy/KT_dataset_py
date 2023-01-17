import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf 

# Load the dataset
df = pd.read_csv("../input/mental-health-in-tech-survey/survey.csv")
df.head()
# To check the shape of dataframe
df.shape
# general summary of the dataframe

df.info()
# Select variables (likely associated with treatment) for analysis

df1  = df[['treatment',"tech_company","Age",'no_employees','family_history','Gender']]

print(df1)
# To check the number of unique values for the object columns
df1.select_dtypes(include=['object']).nunique()
# There are 49 unique values in Gender column. Something is not right!
# To inspect data in the Gender column.

df2 = df1["Gender"].value_counts()
df2
# To purge and group data in Gender column into Nonbinary, Female and Male 

df1["Gender"].replace(['A little about you', 'Agender',
                       'All', 'Androgyne', 'Enby', 'non-binary',
                       'Nah', 'something kinda male?', 'p',
                       'ostensibly male, unsure what that really means',
                       'Genderqueer', 'queer/she/they', 'Neuter', 'Trans woman',
                       'Trans-female', 'queer', 'fluid', 'fluid', 'male leaning androgynous',
                       'Female (trans)', 'Guy (-ish) ^_^'], 'Nonbinary', inplace=True)


df1["Gender"].replace(['Cis Female', 'F', 'Femake', 'Female ', 'Female (cis)', 'Woman',
                       'femail', 'female', 'woman', 'cis-female/femme', 'f'], 'Female', inplace=True)

df1["Gender"].replace(['Cis Male', 'Cis Man', 'M', 'Mail', 'Make', 'Mal',
                       'Male ', 'Male (CIS)', 'Male-ish', 'Man', 'm', 'cis male',
                       'maile','male','msle','Malr'], 'Male', inplace=True)


df1
# Create dummy variables 
df_new = pd.get_dummies(df1, columns = ['treatment', 'tech_company',
                                            'no_employees', 'family_history', 'Gender']) 
df_new
# Drop useless variables
df_new.drop(['tech_company_No', "treatment_No", "family_history_No"], axis = 1,inplace=True) 
df_new
# Compute the correlation matrix
corr = df_new.corr()

# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot = True)
# Calculate the addjusted odds ratio (adjusted with Age) with different independent variables 
list_of_variables = ["Gender_Female","Gender_Nonbinary","family_history_Yes","tech_company_Yes"]

# list of models
models = []

for var in list_of_variables:
    formula = "treatment_Yes ~ " + var + " + Age" 
    models.append(smf.glm(formula=formula, data=df_new, family=sm.families.Binomial()).fit())
    
# To view the summary of individual results ie. 0 = Gender_Female and et cetera
list_inv =[0,1,2,3]

for x in list_inv:
    print(models[x].summary())
    print("odds ratio")
    print(np.exp(models[x].params))
    print()
    print("95% CI")
    print(np.exp(models[x].conf_int()).round(4))
    print()
    print("p-value")
    print(models[x].pvalues)
    print()