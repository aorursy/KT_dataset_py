# Importing Packages



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sb



# Loading the Haberman Data into a pandas DataFrame and adding the column names

# 'Age_of_Patient','Operation_Year','Axil_Nodes','Survival_Status'



haberman_data=pd.read_csv("../input/haberman.csv",header=None, names=['Age_of_Patient','Operation_Year','Axil_Nodes','Survival_Status'])

# Printing the number of data points



print(haberman_data.shape)

# Printing the column names in Haberman Dataset



print(haberman_data.columns)

# Replacing the Survival_Status Values



haberman_data['Survival_Status']=haberman_data['Survival_Status'].replace([1,2],['yes','no'])

print("Survival Status Count of Patients:")

print(haberman_data['Survival_Status'].value_counts())
haberman_data.describe()
# Plotting the Histogram for Age of Patient directly and also  based on Cross Label i.e. Survival Status



plt.hist(haberman_data["Age_of_Patient"])

plt.xlabel("Age of Patient")



sb.FacetGrid(haberman_data,hue='Survival_Status',height=5.5).map(sb.distplot,"Age_of_Patient").add_legend()

plt.show()
# Plotting the Histogram for Operation Year directly and also  based on Cross Label i.e. Survival Status



plt.hist(haberman_data["Operation_Year"])

plt.xlabel("Year  of Operation")



sb.FacetGrid(haberman_data,hue="Survival_Status",height=5.5).map(sb.distplot,"Operation_Year").add_legend()

plt.show()
# Plotting the Histogram for Axil_Nodes directly and also  based on Cross Label i.e. Survival Status



plt.hist(haberman_data["Axil_Nodes"])

plt.xlabel("Axil Nodes")



sb.FacetGrid(haberman_data,hue="Survival_Status",height=5.5).map(sb.distplot,"Axil_Nodes")

plt.show()
# Segregating the data based on the Class Label i.e. Survival Status



haberman_data_Survived=haberman_data.loc[haberman_data["Survival_Status"]=="yes"]

haberman_data_Not_Survived=haberman_data.loc[haberman_data["Survival_Status"]=="no"]

# Plotting the PDF,CDF for Age of Patient for the above Segregated Data



density_age_survived,bin_edges_age_survived=np.histogram(haberman_data_Survived['Age_of_Patient'],bins=10,density=True)

pdf_age_survived=(density_age_survived)/(sum(density_age_survived))





density_Age_Not_Survived,bin_Edges_Age_Not_Survived=np.histogram(haberman_data_Not_Survived['Age_of_Patient'],bins=10,density=True)

pdf_Age_Not_Survived=(density_Age_Not_Survived)/(sum(density_Age_Not_Survived))



print("Bin Edges Survived : {}".format(bin_edges_age_survived))

print("PDF Survived : {}".format(pdf_age_survived))

print("Bin Edges Not Survived :{}".format(bin_Edges_Age_Not_Survived))

print("PDF Not Survived : {}".format(pdf_Age_Not_Survived))





cdf_Age_Not_Survived=np.cumsum(pdf_Age_Not_Survived)

cdf_age_survived=np.cumsum(pdf_age_survived)



plt.plot(bin_edges_age_survived[1:],pdf_age_survived)

plt.plot(bin_edges_age_survived[1:],cdf_age_survived)

plt.plot(bin_Edges_Age_Not_Survived[1:],pdf_Age_Not_Survived)

plt.plot(bin_Edges_Age_Not_Survived[1:],cdf_Age_Not_Survived)

plt.xlabel('Age of Patient')

plt.legend(['Survived_PDF','Survived_CDF','Not Survived PDF','Not Survived CDF'])

plt.show()



# Plotting the PDF,CDF for Year of Operation for the above Segregated Data



density_op_year_survived,bin_edges_op_year_survived=np.histogram(haberman_data_Survived['Operation_Year'],bins=10,density=True)

pdf_op_year_survived=(density_op_year_survived)/(sum(density_op_year_survived))



density_Op_Year_Not_Survived,bin_Edges_Op_Year_Not_Survived=np.histogram(haberman_data_Not_Survived['Operation_Year'],bins=10,density=True)

pdf_Op_Year_Not_Survived=(density_Op_Year_Not_Survived)/(sum(density_Op_Year_Not_Survived))



print("Bin Edges Survived : {}".format(bin_edges_op_year_survived))

print("PDF Survived : {}".format(pdf_op_year_survived))

print("Bin Edges Not Survived :{}".format(bin_Edges_Op_Year_Not_Survived))

print("PDF Not Survived : {}".format(pdf_Op_Year_Not_Survived))



cdf_op_year_survived=np.cumsum(pdf_op_year_survived)

cdf_Op_Year_Not_Survived=np.cumsum(pdf_Op_Year_Not_Survived)





plt.plot(bin_edges_op_year_survived[1:],pdf_op_year_survived)

plt.plot(bin_edges_op_year_survived[1:],cdf_op_year_survived)

plt.plot(bin_Edges_Op_Year_Not_Survived[1:],pdf_Op_Year_Not_Survived)

plt.plot(bin_Edges_Op_Year_Not_Survived[1:],cdf_Op_Year_Not_Survived)

plt.xlabel('Operation Year')

plt.legend(['Survived_PDF','Survived_CDF','Not Survived PDF','Not Survived CDF'])

plt.show()
# Plotting the PDF,CDF for Axil Nodes for the above Segregated Data



density_axil_nodes_survived,bin_edges_axil_nodes_survived=np.histogram(haberman_data_Survived['Axil_Nodes'],bins=10,density=True)

pdf_axil_nodes_survived=(density_axil_nodes_survived)/(sum(density_axil_nodes_survived))



density_Axil_Nodes_Not_Survived,bin_Edges_Axil_Nodes_Not_Survived=np.histogram(haberman_data_Not_Survived['Axil_Nodes'],bins=10,density=True)

pdf_Axil_Nodes_Not_Survived=(density_Axil_Nodes_Not_Survived)/(sum(density_Axil_Nodes_Not_Survived))



print("Bin Edges Survived : {}".format(bin_edges_axil_nodes_survived))

print("PDF Survived : {}".format(pdf_axil_nodes_survived))

print("Bin Edges Not Survived :{}".format(bin_Edges_Axil_Nodes_Not_Survived))

print("PDF Not Survived : {}".format(pdf_Axil_Nodes_Not_Survived))







cdf_axil_nodes_survived=np.cumsum(pdf_axil_nodes_survived)

cdf_Axil_Nodes_Not_Survived=np.cumsum(pdf_Axil_Nodes_Not_Survived)





plt.plot(bin_edges_axil_nodes_survived[1:],pdf_axil_nodes_survived)

plt.plot(bin_edges_axil_nodes_survived[1:],cdf_axil_nodes_survived)

plt.plot(bin_Edges_Axil_Nodes_Not_Survived[1:],pdf_Axil_Nodes_Not_Survived)

plt.plot(bin_Edges_Axil_Nodes_Not_Survived[1:],cdf_Axil_Nodes_Not_Survived)

plt.xlabel('Axil Nodes')

plt.legend(['Survived_PDF','Survived_CDF','Not Survived PDF','Not Survived CDF'])

plt.show()
sb.boxplot(x="Survival_Status",y="Operation_Year",data=haberman_data)

plt.show()
sb.boxplot(x="Survival_Status",y="Age_of_Patient",data=haberman_data)

plt.show()
sb.boxplot(x="Survival_Status",y="Axil_Nodes",data=haberman_data)

plt.show()
sb.violinplot(x="Survival_Status",y="Age_of_Patient",data=haberman_data)

plt.show()
sb.violinplot(x="Survival_Status",y="Axil_Nodes",data=haberman_data)

plt.show()
sb.violinplot(x="Survival_Status",y="Operation_Year",data=haberman_data)

plt.show()
sb.pairplot(haberman_data,hue="Survival_Status",height=4)

plt.show()
sb.set_style('whitegrid')

sb.FacetGrid(haberman_data,hue='Survival_Status',height=5).map(plt.scatter,'Axil_Nodes','Age_of_Patient')

plt.show()


sb.jointplot(x="Axil_Nodes",y="Age_of_Patient",data=haberman_data_Survived,kind="kde",color='g')

sb.jointplot(x="Axil_Nodes",y="Age_of_Patient",data=haberman_data_Not_Survived,kind="kde",color='r')

plt.show()
