#importing the packages 

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

from matplotlib.colors import ListedColormap

import seaborn as sns

print("Setup Complete")
#loading the dataset and adding the column header

cancer_data = pd.read_csv('../input/breast_cancer_prognostic.csv', names =["id","diagnosis","time","radius_mean","texture_mean",

                                                    "perimeter_mean","area_mean","smoothness_mean","compactness_mean",

                                                    "concavity_mean","concavityPoints_mean","symmetry_mean","fractal_mean",

                                                    "radius_standard_error","texture_standard_error","perimeter_standard_error",

                                                    "area_standard_error","smoothness_standard_error","compactness_standard_error",

                                                    "concavity_standard_error","concavityPoints_standard_error","symmetry_standard_error",

                                                    "fractal_standard_error","radius_worst","texture_worst","perimeter_worst","area_worst",

                                                    "smoothness_worst","compactness_worst","concavity_worst","concavityPoints_worst","symmetry_worst",

                                                    "fractal_worst","tumor_size","lymph_node"])
cancer_data.head()
print(cancer_data.info())
#breast cancer diagnosis size

print(cancer_data.groupby('diagnosis').size())
g = sns.pairplot(cancer_data, vars=["time","radius_mean","texture_mean",

                                                    "perimeter_mean","area_mean","smoothness_mean","compactness_mean",

                                                    "concavity_mean","concavityPoints_mean","symmetry_mean","fractal_mean",

                                            "tumor_size"], hue ="diagnosis",palette=sns.color_palette(['#DC143C','#DAA520']))
sns.boxplot(x=cancer_data['diagnosis'], y=cancer_data['time'],palette=sns.color_palette(['#DC143C','#DAA520']))
sns.boxplot(x=cancer_data['diagnosis'], y=cancer_data['radius_mean'],palette=sns.color_palette(['#DC143C','#DAA520']))
sns.boxplot(x=cancer_data['diagnosis'], y=cancer_data['texture_mean'],palette=sns.color_palette(['#DC143C','#DAA520']))
sns.boxplot(x=cancer_data['diagnosis'], y=cancer_data['perimeter_mean'],palette=sns.color_palette(['#DC143C','#DAA520']))
sns.boxplot(x=cancer_data['diagnosis'], y=cancer_data['area_mean'],palette=sns.color_palette(['#DC143C','#DAA520']))
sns.boxplot(x=cancer_data['diagnosis'], y=cancer_data['smoothness_mean'],palette=sns.color_palette(['#DC143C','#DAA520']))
sns.boxplot(x=cancer_data['diagnosis'], y=cancer_data['compactness_mean'],palette=sns.color_palette(['#DC143C','#DAA520']))
sns.boxplot(x=cancer_data['diagnosis'], y=cancer_data['concavity_mean'],palette=sns.color_palette(['#DC143C','#DAA520']))
sns.boxplot(x=cancer_data['diagnosis'], y=cancer_data['concavityPoints_mean'],palette=sns.color_palette(['#DC143C','#DAA520']))
sns.boxplot(x=cancer_data['diagnosis'], y=cancer_data['symmetry_mean'],palette=sns.color_palette(['#DC143C','#DAA520']))
sns.boxplot(x=cancer_data['diagnosis'], y=cancer_data['fractal_mean'],palette=sns.color_palette(['#DC143C','#DAA520']))
sns.boxplot(x=cancer_data['diagnosis'], y=cancer_data['tumor_size'],palette=sns.color_palette(['#DC143C','#DAA520']))
sns.scatterplot(x=cancer_data["radius_mean"],y=cancer_data["perimeter_mean"], palette=sns.color_palette(['#DC143C','#FFD700']),hue=cancer_data["diagnosis"])
sns.scatterplot(x=cancer_data["area_mean"],y=cancer_data["radius_mean"],palette=sns.color_palette(['#DC143C','#FFD700']), hue=cancer_data["diagnosis"])
sns.scatterplot(x=cancer_data["fractal_worst"],y=cancer_data["smoothness_worst"],palette=sns.color_palette(['#DC143C','#FFD700']), hue=cancer_data["diagnosis"])
sns.scatterplot(x=cancer_data["area_standard_error"],y=cancer_data["concavity_mean"], palette=sns.color_palette(['#DC143C','#FFD700']),hue=cancer_data["diagnosis"])
sns.scatterplot(x=cancer_data["area_mean"],y=cancer_data["smoothness_standard_error"],palette=sns.color_palette(['#DC143C','#FFD700']), hue=cancer_data["diagnosis"])
sns.scatterplot(x=cancer_data["radius_standard_error"],y=cancer_data["smoothness_standard_error"],palette=sns.color_palette(['#DC143C','#FFD700']), hue=cancer_data["diagnosis"])
sns.distplot(a=cancer_data['time'],color='DarkRed',vertical=False)

#variable time is slightly skewed and removing few outliers will help achiever the normal distribution

            
sns.distplot(a=cancer_data['radius_standard_error'],color='DarkRed',vertical=False)



#Left skewness observerd for radius standard error to left
sns.distplot(a=cancer_data['texture_standard_error'],color='DarkRed',vertical=False)

#texture standard error is slightly skewed to the left
sns.distplot(a=cancer_data['perimeter_standard_error'],color='DarkRed',vertical=False)

#perimeter standard error is slightly skewed to the left
sns.distplot(a=cancer_data['area_standard_error'],color='DarkRed',vertical=False)

#area standard error is slightly skewed  to the left
sns.distplot(a=cancer_data['smoothness_standard_error'],color='DarkRed',vertical=False)

#smoothness standard error is slightly skewed 
sns.distplot(a=cancer_data['compactness_standard_error'],color='DarkRed',vertical=False)

#compactness standard error is slightly skewed 
sns.distplot(a=cancer_data['concavity_standard_error'],color='DarkRed',vertical=False)

#concavity standard error is slightly skewed 
sns.distplot(a=cancer_data['concavityPoints_standard_error'],color='DarkRed',vertical=False)

#concavity standard error is normal distributed 
sns.distplot(a=cancer_data['symmetry_standard_error'],color='DarkRed',vertical=False)

#symmetry standard error is slightly skewed 
sns.distplot(a=cancer_data['fractal_standard_error'],color='DarkRed',vertical=False)

#fractal standard error is slightly skewed 