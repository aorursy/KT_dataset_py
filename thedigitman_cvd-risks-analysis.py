# Setup Jovian
!pip install jovian --upgrade --quiet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import jovian
#Import the dataset as a Pandas Dataframe
raw_df = pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
#View metadata about the data
print("Shape: ",raw_df.shape, "\n")
print(raw_df.info())
print("\n Any Missing Values: ",raw_df.isnull().values.any())
# Use only data of patients that remained part of the study
data_df = raw_df[raw_df.DEATH_EVENT == 1]
num_records = data_df.count()
# View statistics of the age distribution
data_df.age.describe()
# View statistics of the gender distribution
print(data_df.sex.value_counts())
data_df.sex.value_counts()/96
data_df.anaemia.value_counts()/96
data_df.diabetes.value_counts()/96
data_df.high_blood_pressure.value_counts()/96
data_df.smoking.value_counts()/96
# Determine number of patients not having risk factors above
no_risks_df = data_df.query('anaemia==0 and diabetes==0 and high_blood_pressure==0 and smoking==0')
print(no_risks_df.age.count())
print(no_risks_df.age.count()/96)

# Using a sctterplot, see the distribution of patients having and not having the risk factors. Each risk factor is desplaced from the others to ease visualizing
plt.figure(figsize=(15,8))
plt.scatter(data_df.time,data_df.diabetes)
plt.scatter(data_df.time,data_df.high_blood_pressure+0.2)
plt.scatter(data_df.time,data_df.anaemia+0.4)
plt.scatter(data_df.time,data_df.smoking+0.6)

#Plot a Pareto Diagram

#Create histogram bins
hist_count, hist_bins = np.histogram(data_df.time, bins=[i for i in range(7,246,7)])

plt.figure(figsize=(15,8))
#Plot the histogram bars
plt.bar(hist_bins[:-1], hist_count, width=7)

#Create the percentage of total data for the line plot
lineplot = [] 
lineplot.append(hist_count[0] / hist_count.sum()) #Create the first value
for i in range(1,len(hist_count)):
    lineplot.append(lineplot[i - 1] + (hist_count[i] / hist_count.sum())) #Add the values of each bin to the data.
# Plot the pecentage of total line. We have to multiply the lineplot by the max of count to scale the line plot to the y axis.
plt.plot(hist_bins[:-1],np.dot(lineplot,hist_count.max()), color='red')
print("Cummalative Precentage of Total of each Bin:")
for i in range(len(lineplot)):
    print(f"Bin {i + 1:2} - {100 * lineplot[i]:.2f}%")
# Box plot the time data
data_df.time.plot.box()
#Plot a scatterplot
plt.figure(figsize=(5,5))
plt.scatter(data_df.creatinine_phosphokinase, data_df.time, color="green")
plt.show()
#Plot a scatterplot
plt.figure(figsize=(5,5))
plt.scatter(data_df.ejection_fraction, data_df.time, color="red")
plt.show()
#Plot a scatterplot
plt.figure(figsize=(5,5))
plt.scatter(data_df.platelets, data_df.time, color="yellow")
plt.show()
#Plot a scatterplot
plt.figure(figsize=(5,5))
plt.scatter(data_df.serum_creatinine, data_df.time, color="orange")
plt.show()
#Plot a scatterplot
plt.figure(figsize=(5,5))
plt.scatter(data_df.serum_sodium, data_df.time, color="blue")
plt.show()
#135-145
# Calculate the Standard Deviation of serum_sodium
data_df.serum_sodium.std()
jovian.commit(project="CVDRisks")
