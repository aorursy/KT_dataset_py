#importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#loading data to pandas dataframe
data = pd.read_csv('/content/haberman.csv')

#printing first 10 rows
print(data.head(10)) 
print('\n')
print(data.columns, end = '\n\n')
print(data.shape, end = '\n\n')

#renaming columns
#data.columns({'Age', 'Oper_year', 'Axil_nodes', 'Survival_status'}, inplace = True)
data = data.rename(columns={'30': "Age", '64': "Oper_year", '1': "Axil_nodes", '1.1': "Survival_status"}) #https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rename.html
print(data.head(10))
#unique values and its count of Survival_status column
data['Survival_status'].value_counts()
# 1 = the patient survived 5 years or longer
# 2 = the patient died within 5 year
survived = data.loc[data['Survival_status'] == 1]
not_survived = data.loc[data['Survival_status'] == 2]

print(type(survived))
print(survived.head(10))

print(type(not_survived))
print(not_survived.head(10))
plt.plot(survived["Age"], np.zeros_like(survived['Age']), 'o', label = 'Survived') # https://numpy.org/doc/stable/reference/generated/numpy.zeros_like.html
plt.plot(not_survived["Age"], np.zeros_like(not_survived['Age']), 'o', label = 'Not_survived') 
#plt.legend('Survived', 'Not_survived')

#adding axis label and display label 
plt.xlabel('Age') 
plt.ylabel('Array with zeros') 
plt.legend()  #https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.legend.html

plt.grid() #to display grids in the output
plt.show() #to display plot(output)
plt.plot(survived['Oper_year'], np.zeros_like(survived['Oper_year']), '*', label = 'Survived') 
plt.plot(not_survived['Oper_year'], np.zeros_like(not_survived['Oper_year']), '^', label = 'Not_survived') 


#adding axis label and display label 
plt.xlabel('Year of operation') 
plt.ylabel('Array with zeros') 
plt.legend()  

plt.grid()
plt.show() 
plt.plot(survived['Axil_nodes'], np.zeros_like(survived['Axil_nodes']), '*', label = 'Survived') 
plt.plot(not_survived['Axil_nodes'], np.zeros_like(not_survived['Axil_nodes']), '^', label = 'Not_survived') 


#adding axis label and display label 
plt.xlabel('Number of auxiliary nodes') 
plt.ylabel('Array with zeros') 
plt.legend()  

plt.grid()
plt.show() 
plt.hist(survived["Age"], bins = 100, label = 'Survived')
#plt.hist(survived["Age"],  label = 'Not_survived')
plt.legend()  
plt.show()
#data.plot(survived['Age'], survived['Oper_year'], 'rs', not_survived['Age'], not_survived['Oper_year'], 'b*')
print('Plot with age and oper_year')
data.plot(kind='scatter', x='Age', y='Oper_year')
plt.grid()
plt.show()

print('Plot with age and auxiliary node')
data.plot(kind='scatter', x='Age', y='Axil_nodes')
plt.grid()
plt.show()

print('Plot with auxiliary nodes and oper_year')
data.plot(kind='scatter', x='Axil_nodes', y='Oper_year')
plt.grid()
plt.show()
#plotting two attributes in a graph by differtiating survival_status; survived = red, not_survived = blue

print('Plot with age and oper_year')
for n in range(0, 305):
  if data['Survival_status'][n] == 1:
    line1 = plt.scatter(data['Age'][n], data['Oper_year'][n], color = 'red')
  else:
    line2 = plt.scatter(data['Age'][n], data['Oper_year'][n], color = 'blue')
plt.xlabel('Age')
plt.ylabel('Oper_year')  
plt.legend((line1, line2), ('survived', 'not_survived'))
#plt.legend()
plt.grid()
plt.show()


print('Plot with age and auxiliary node')
for n in range(0, 305):
  if data['Survival_status'][n] == 1:
    line1 = plt.scatter(data['Age'][n], data['Axil_nodes'][n], color = 'red')
  else:
    line2 = plt.scatter(data['Age'][n], data['Axil_nodes'][n], color = 'blue')
plt.xlabel('Age')
plt.ylabel('Axil_nodes')  
plt.legend((line1, line2), ('survived', 'not_survived'))
#plt.legend()
plt.grid()
plt.show()


print('Plot with auxiliary nodes and oper_year')
for n in range(0, 305):
  if data['Survival_status'][n] == 1:
    line1 = plt.scatter(data['Axil_nodes'][n], data['Oper_year'][n], color = 'red')
  else:
    line2 = plt.scatter(data['Axil_nodes'][n], data['Oper_year'][n], color = 'blue')
plt.xlabel('Axil_nodes')
plt.ylabel('Oper_year')  
plt.legend((line1, line2), ('survived', 'not_survived'))
#plt.legend()
plt.grid()
plt.show()
#mean
print('Mean age of survived patients: ', np.mean(survived['Age']))
print('Mean of operation year survived patients: ', np.mean(survived['Oper_year']))
print('Mean affected nodes of survived patients: ', np.mean(survived['Axil_nodes']))

print('Mean age of not survived patients: ', np.mean(not_survived['Age']))
print('Mean of operation year not survived patients: ', np.mean(not_survived['Oper_year']))
print('Mean affected nodes of not survived patients: ', np.mean(not_survived['Axil_nodes']))
#standard deviation
print('SD age of survived patients: ', np.std(survived['Age']))
print('SD of operation year survived patients: ', np.std(survived['Oper_year']))
print('SD affected nodes of survived patients: ', np.std(survived['Axil_nodes']))

print('SD age of not survived patients: ', np.std(not_survived['Age']))
print('SD of operation year not survived patients: ', np.std(not_survived['Oper_year']))
print('SD affected nodes of not survived patients: ', np.std(not_survived['Axil_nodes']))
#Median
print('Median age of survived patients: ', np.median(survived['Age']))
print('Median of operation year survived patients: ', np.median(survived['Oper_year']))
print('Median affected nodes of survived patients: ', np.median(survived['Axil_nodes']))

print('Median age of not survived patients: ', np.median(not_survived['Age']))
print('Median of operation year not survived patients: ', np.median(not_survived['Oper_year']))
print('Median affected nodes of not survived patients: ', np.median(not_survived['Axil_nodes']))


#Quantiles 
print('Quantiles of age of survived patients: ', np.percentile(survived['Age'], np.arange(1, 101, 25)))
print('Quantiles of operation year survived patients: ', np.percentile(survived['Oper_year'], np.arange(1, 101, 25)))
print('Quantiles of affected nodes of survived patients: ', np.percentile(survived['Axil_nodes'], np.arange(1, 101, 25)))

print('Quantiles of age of not survived patients: ', np.percentile(not_survived['Age'], np.arange(1, 101, 25)))
print('Quantiles of operation year not survived patients: ', np.percentile(not_survived['Oper_year'], np.arange(1, 101, 25)))
print('Quantiles of affected nodes of not survived patients: ', np.percentile(not_survived['Axil_nodes'], np.arange(1, 101, 25)))



#Percentiles 
print('90th %tile of age for survived patients: ', np.percentile(survived['Age'], 90))
print('90th %tile of operation year for survived patients: ', np.percentile(survived['Oper_year'], 90))
print('90th %tile of affected nodes for survived patients: ', np.percentile(survived['Axil_nodes'], 90))

print('90th %tile of age for survived patients: ', np.percentile(not_survived['Age'], 90))
print('90th %tile of operation year not survived patients: ', np.percentile(not_survived['Oper_year'], 90))
print('90th %tile of affected nodes for not survived patients: ', np.percentile(not_survived['Axil_nodes'], 90))

#whisker plot
sns.boxplot(x = 'Survival_status', y = 'Age', data = data)
plt.show()
sns.boxplot(x = 'Survival_status', y = 'Oper_year', data = data)
plt.show()
sns.boxplot(x = 'Survival_status', y = 'Axil_nodes', data = data)
plt.show()

#violin plot
sns.violinplot(x = "Survival_status", y = "Age", data = data, size = 90)
plt.show()
sns.violinplot(x = "Survival_status", y = "Oper_year", data = data, size = 90)
plt.show()
sns.violinplot(x = "Survival_status", y = "Axil_nodes", data = data, size = 90)
plt.show()
sns.FacetGrid(data, hue = "Survival_status", height = 5).map(sns.distplot, "Age").add_legend()
plt.ylabel('Variance')
sns.FacetGrid(data, hue = "Survival_status", height = 5).map(sns.distplot, "Oper_year").add_legend()
plt.ylabel('Variance')
sns.FacetGrid(data, hue = "Survival_status", height = 5).map(sns.distplot, "Axil_nodes").add_legend()
plt.ylabel('Variance')
plt.show()
# plotting cdf and pdf by age feature and survival status
counts, bin_edges = np.histogram(survived['Age'], bins=10, density = True)                               
pdf = counts/(sum(counts))
print(pdf)
print(bin_edges)
cdf = np.cumsum(pdf)
plt.xlabel('Age')
plt.plot(bin_edges[1:],pdf, label = 'survived_pdf')
plt.plot(bin_edges[1:], cdf, label = 'survived_cdf')

# plotting cdf and pdf by age feature and non survival status
counts, bin_edges = np.histogram(not_survived['Age'], bins=10, density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label = 'non_survived_pdf')
plt.plot(bin_edges[1:], cdf, label = 'non_survived_cdf')
plt.legend()

plt.show();
# plotting cdf and pdf by year of operation feature and survival status
counts, bin_edges = np.histogram(survived['Oper_year'], bins=10, density = True)                               
pdf = counts/(sum(counts))
print(pdf)
print(bin_edges)
cdf = np.cumsum(pdf)
plt.xlabel('Oper_year')
plt.plot(bin_edges[1:],pdf, label = 'survived_pdf')
plt.plot(bin_edges[1:], cdf, label = 'survived_cdf')

# plotting cdf and pdf by year of operation feature and non survival status
counts, bin_edges = np.histogram(not_survived['Oper_year'], bins=10, density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label = 'non_survived_pdf')
plt.plot(bin_edges[1:], cdf, label = 'non_survived_cdf')
plt.legend()

plt.show();
# plotting cdf and pdf by auxiliary nodes feature and survival status
counts, bin_edges = np.histogram(survived['Axil_nodes'], bins=10, density = True)                               
pdf = counts/(sum(counts))
print(pdf)
print(bin_edges)
cdf = np.cumsum(pdf)
plt.xlabel('Axil_nodes')
plt.plot(bin_edges[1:],pdf, label = 'survived_pdf')
plt.plot(bin_edges[1:], cdf, label = 'survived_cdf')

# plotting cdf and pdf by auxiliary nodes feature and non survival status
counts, bin_edges = np.histogram(not_survived['Axil_nodes'], bins=10, density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label = 'non_survived_pdf')
plt.plot(bin_edges[1:], cdf, label = 'non_survived_cdf')
plt.legend()

plt.show();
