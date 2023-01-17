import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
student_performance=pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv')
student_performance.dtypes
student_performance.head()
student_performance.shape
student_performance.tail()
student_performance.columns
student_performance.info()
student_performance.describe()
student_performance.isnull()
student_performance.isnull().any()
#shows for each column the percentage of null values 
student_performance.isnull().sum() / student_performance.shape[0]
student_performance['parental level of education'].value_counts()
student_performance['race/ethnicity'].value_counts()
student_performance['gender'].value_counts()
student_performance['lunch'].value_counts()
student_performance['test preparation course'].value_counts()
student_performance.iloc[:, 5:8].sum(axis=1)
student_performance['total_marks']= student_performance.iloc[:, 5:8].sum(axis=1)
print(student_performance)
#2-D scatter plot:
#ALWAYS understand the axis: labels and scale.

student_performance.plot(kind='scatter', x='total_marks', y='writing score') ;
plt.show()

#cannot make much sense out it. 
#What if we color the points by thier class-label/flower-type.
# 2-D Scatter plot with color-coding for each flower type/class.
# Here 'sns' corresponds to seaborn. 
sns.set_style("darkgrid");
sns.FacetGrid(student_performance, hue="lunch", size=4) \
   .map(plt.scatter, "total_marks", "math score") \
   .add_legend();
#plt.show();

# Notice that the blue points can be easily seperated 
# from red and green by drawing a line. 
# But red and green data points cannot be easily seperated.
# Can we draw multiple 2-D scatter plots for each combination of features?
# How many cobinations exist? 4C2 = 6.
# 2-D Scatter plot with color-coding for each flower type/class.
# Here 'sns' corresponds to seaborn. 
sns.set_style("darkgrid");
sns.FacetGrid(student_performance, hue="parental level of education", size=4) \
   .map(plt.scatter, "total_marks", "math score") \
   .add_legend();
#plt.show();

# Notice that the blue points can be easily seperated 
# from red and green by drawing a line. 
# But red and green data points cannot be easily seperated.
# Can we draw multiple 2-D scatter plots for each combination of features?
# How many cobinations exist? 4C2 = 6.
# 2-D Scatter plot with color-coding for each flower type/class.
# Here 'sns' corresponds to seaborn. 
sns.set_style("darkgrid");
sns.FacetGrid(student_performance, hue="gender", size=4) \
   .map(plt.scatter, "total_marks", "math score") \
   .add_legend();
#plt.show();

# Notice that the blue points can be easily seperated 
# from red and green by drawing a line. 
# But red and green data points cannot be easily seperated.
# Can we draw multiple 2-D scatter plots for each combination of features?
# How many cobinations exist? 4C2 = 6.
# 2-D Scatter plot with color-coding for each flower type/class.
# Here 'sns' corresponds to seaborn. 
sns.set_style("darkgrid");
sns.FacetGrid(student_performance, hue="race/ethnicity", size=4) \
   .map(plt.scatter, "total_marks", "math score") \
   .add_legend();
#plt.show();

# Notice that the blue points can be easily seperated 
# from red and green by drawing a line. 
# But red and green data points cannot be easily seperated.
# Can we draw multiple 2-D scatter plots for each combination of features?
# How many cobinations exist? 4C2 = 6.
# 2-D Scatter plot with color-coding for each flower type/class.
# Here 'sns' corresponds to seaborn. 
sns.set_style("darkgrid");
sns.FacetGrid(student_performance, hue="test preparation course", size=4) \
   .map(plt.scatter, "total_marks", "math score") \
   .add_legend();
#plt.show();

# Notice that the blue points can be easily seperated 
# from red and green by drawing a line. 
# But red and green data points cannot be easily seperated.
# Can we draw multiple 2-D scatter plots for each combination of features?
# How many cobinations exist? 4C2 = 6.
# pairwise scatter plot: Pair-Plot
# Dis-advantages: 
##Cannot be used when number of features are high.
##Cannot visualize higher dimensional patterns in 3-D and 4-D. 
#Only possible to view 2D patterns.
plt.close();
sns.set_style("darkgrid");
sns.pairplot(student_performance, hue="gender", size=3).add_legend();
#plt.show()
# NOTE: the diagnol elements are PDFs for each feature. PDFs are expalined below.
# pairwise scatter plot: Pair-Plot
# Dis-advantages: 
##Cannot be used when number of features are high.
##Cannot visualize higher dimensional patterns in 3-D and 4-D. 
#Only possible to view 2D patterns.
plt.close();
sns.set_style("darkgrid");
sns.pairplot(student_performance, hue="race/ethnicity", size=3).add_legend();
#plt.show()
# NOTE: the diagnol elements are PDFs for each feature. PDFs are expalined below.
# pairwise scatter plot: Pair-Plot
# Dis-advantages: 
##Cannot be used when number of features are high.
##Cannot visualize higher dimensional patterns in 3-D and 4-D. 
#Only possible to view 2D patterns.
plt.close();
sns.set_style("darkgrid");
sns.pairplot(student_performance, hue="parental level of education", size=3).add_legend();
#plt.show()
# NOTE: the diagnol elements are PDFs for each feature. PDFs are expalined below.
# pairwise scatter plot: Pair-Plot
# Dis-advantages: 
##Cannot be used when number of features are high.
##Cannot visualize higher dimensional patterns in 3-D and 4-D. 
#Only possible to view 2D patterns.
plt.close();
sns.set_style("darkgrid");
sns.pairplot(student_performance, hue="lunch", size=3).add_legend();
#plt.show()
# NOTE: the diagnol elements are PDFs for each feature. PDFs are expalined below.
# pairwise scatter plot: Pair-Plot
# Dis-advantages: 
##Cannot be used when number of features are high.
##Cannot visualize higher dimensional patterns in 3-D and 4-D. 
#Only possible to view 2D patterns.
plt.close();
sns.set_style("darkgrid");
sns.pairplot(student_performance, hue="test preparation course", size=3).add_legend();
#plt.show()
# NOTE: the diagnol elements are PDFs for each feature. PDFs are expalined below.
student_performance[student_performance.total_marks==student_performance.total_marks.max()]
student_performance[student_performance.total_marks==student_performance.total_marks.min()]
student_performance.sort_values(by=['total_marks']).head(10)
student_performance.sort_values(by=['total_marks']).tail(10)
student_privilege=student_performance.groupby('lunch')
pd.options.display.max_columns = 4000
print(student_privilege.describe())
student_gender=student_performance.groupby('gender')
print(student_gender.describe())
under_privileged=student_performance[student_performance['lunch']=='free/reduced']
under_privileged.head()
under_privilegedloc=student_performance.loc[student_performance['lunch']=='free/reduced']
under_privilegedloc.head()
under_privileged[(under_privileged['math score']>=40) & (under_privileged['reading score']>=40) & (under_privileged['writing score']>=40)].shape
under_privileged[(under_privileged['math score']<40) | (under_privileged['reading score']<40) | (under_privileged['writing score']<40)].shape
pass_percentage_of_under_privileged_students=((under_privileged[(under_privileged['math score']>=40) & (under_privileged['reading score']>=40) & (under_privileged['writing score']>=40)].shape)[0])/((under_privileged.shape)[0])
pass_percentage_of_under_privileged_students
under_privileged['gender'].value_counts()
pass_percentage_of_under_privileged_girls=((under_privileged[(under_privileged['math score']>=40) & (under_privileged['reading score']>=40) & (under_privileged['writing score']>=40) & (under_privileged['gender']== 'female')].shape)[0])/((under_privileged[under_privileged['gender']== 'female'].shape)[0])
pass_percentage_of_under_privileged_girls
pass_percentage_of_under_privileged_boys=((under_privileged[(under_privileged['math score']>=40) & (under_privileged['reading score']>=40) & (under_privileged['writing score']>=40) & (under_privileged['gender']== 'male')].shape)[0])/((under_privileged[under_privileged['gender']== 'male'].shape)[0])
pass_percentage_of_under_privileged_boys
privileged=student_performance[student_performance['lunch']=='standard']
privileged.shape
privileged[(privileged['math score']>=40) & (privileged['reading score']>=40) & (privileged['writing score']>=40)].shape
privileged[(privileged['math score']<40) | (privileged['reading score']<40) | (privileged['writing score']<40)].shape
pass_percentage_of_privileged_students=((privileged[(privileged['math score']>=40) & (privileged['reading score']>=40) & (privileged['writing score']>=40)].shape)[0])/((privileged.shape)[0])
pass_percentage_of_privileged_students
privileged['gender'].value_counts()
pass_percentage_of_privileged_girls=((privileged[(privileged['math score']>=40) & (privileged['reading score']>=40) & (privileged['writing score']>=40) & (privileged['gender']== 'female')].shape)[0])/((privileged[privileged['gender']== 'female'].shape)[0])
pass_percentage_of_privileged_girls
pass_percentage_of_privileged_boys=((privileged[(privileged['math score']>=40) & (privileged['reading score']>=40) & (privileged['writing score']>=40) & (privileged['gender']== 'male')].shape)[0])/((privileged[privileged['gender']== 'male'].shape)[0])
pass_percentage_of_privileged_boys
students_with_course=student_performance[student_performance['test preparation course']=='completed']
students_with_course.shape
students_with_course.lunch.value_counts()
students_with_course[students_with_course['lunch']=='free/reduced'].shape
pass_percentage_of_students_with_course=((students_with_course[(students_with_course['math score']>=40) & (students_with_course['reading score']>=40) & (students_with_course['writing score']>=40)].shape)[0])/((students_with_course.shape)[0])
pass_percentage_of_students_with_course
pass_percentage_of_privileged_students_with_course=((students_with_course[(students_with_course['math score']>=40) & (students_with_course['reading score']>=40) & (students_with_course['writing score']>=40) & (students_with_course['lunch']=='standard')].shape)[0])/((students_with_course[students_with_course['lunch']=='standard'].shape)[0])
pass_percentage_of_privileged_students_with_course
pass_percentage_of_unprivileged_students_with_course=((students_with_course[(students_with_course['math score']>=40) & (students_with_course['reading score']>=40) & (students_with_course['writing score']>=40) & (students_with_course['lunch']=='free/reduced')].shape)[0])/((students_with_course[students_with_course['lunch']=='free/reduced'].shape)[0])
pass_percentage_of_unprivileged_students_with_course
students_with_course.gender.value_counts()
pass_percentage_of_boys_with_course=((students_with_course[(students_with_course['math score']>=40) & (students_with_course['reading score']>=40) & (students_with_course['writing score']>=40) & (students_with_course['gender']== 'male')].shape)[0]) /((students_with_course[students_with_course['gender']== 'male'].shape)[0])
pass_percentage_of_boys_with_course
pass_percentage_of_girls_with_course=((students_with_course[(students_with_course['math score']>=40) & (students_with_course['reading score']>=40) & (students_with_course['writing score']>=40) & (students_with_course['gender']== 'female')].shape)[0]) /((students_with_course[students_with_course['gender']== 'female'].shape)[0])
pass_percentage_of_girls_with_course
students_without_course=student_performance[student_performance['test preparation course']=='none']
students_without_course.shape
students_without_course[students_without_course['lunch']=='standard'].shape
students_without_course[students_without_course['lunch']=='free/reduced'].shape
pass_percentage_of_students_without_course=((students_without_course[(students_without_course['math score']>=40) & (students_without_course['reading score']>=40) & (students_without_course['writing score']>=40)].shape)[0])/((students_without_course.shape)[0])
pass_percentage_of_students_without_course
pass_percentage_of_privileged_students_without_course=((students_without_course[(students_without_course['math score']>=40) & (students_without_course['reading score']>=40) & (students_without_course['writing score']>=40) & (students_without_course['lunch']=='standard')].shape)[0])/((students_without_course[students_without_course['lunch']=='standard'].shape)[0])
pass_percentage_of_privileged_students_without_course
pass_percentage_of_unprivileged_students_without_course=((students_without_course[(students_without_course['math score']>40) & (students_without_course['reading score']>40) & (students_without_course['writing score']>40) & (students_without_course['lunch']=='free/reduced')].shape)[0])/((students_without_course[students_without_course['lunch']=='free/reduced'].shape)[0])
pass_percentage_of_unprivileged_students_without_course
pass_percentage_of_boys_without_course=((students_without_course[(students_without_course['math score']>=40) & (students_without_course['reading score']>=40) & (students_without_course['writing score']>=40) & (students_without_course['gender']== 'male')].shape)[0]) /((students_without_course[students_without_course['gender']== 'male'].shape)[0])
pass_percentage_of_boys_without_course
pass_percentage_of_girls_without_course=((students_without_course[(students_without_course['math score']>=40) & (students_without_course['reading score']>=40) & (students_without_course['writing score']>=40) & (students_without_course['gender']== 'female')].shape)[0]) /((students_without_course[students_without_course['gender']== 'female'].shape)[0])
pass_percentage_of_girls_without_course
student_performance.columns
sns.FacetGrid(student_performance, hue="gender", size=5) \
   .map(sns.distplot, "total_marks") \
   .add_legend();
plt.show();
sns.FacetGrid(student_performance, hue="race/ethnicity", size=5) \
   .map(sns.distplot, "total_marks") \
   .add_legend();
plt.show();
sns.FacetGrid(student_performance, hue="parental level of education", size=5) \
   .map(sns.distplot, "total_marks") \
   .add_legend();
plt.show();
sns.FacetGrid(student_performance, hue="lunch", size=5) \
   .map(sns.distplot, "total_marks") \
   .add_legend();
plt.show();
sns.FacetGrid(student_performance, hue="test preparation course", size=5) \
   .map(sns.distplot, "total_marks") \
   .add_legend();
plt.show();
#REFER program.txt
# Need for Cumulative Distribution Function (CDF)
# We can visually see what percentage of versicolor flowers have a 
# petal_length of less than 5?
# How to construct a CDF?
# How to read a CDF?

#Plot CDF of petal_length
#print(iris_setosa['petal.length'])
print(np.histogram(student_performance['math score'], bins=10, 
                                 density = False))
counts, bin_edges = np.histogram(student_performance['math score'], bins=10, 
                                 density = True)
#print(max(iris_setosa['petal.length']))
#print(min(iris_setosa['petal.length']))
#print("counts:",counts)
#print("Sum:",sum(counts))
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)


counts, bin_edges = np.histogram(student_performance['math score'], bins=5, 
                                 density = True)
#print (counts)
#print (bin_edges)
pdf = counts/(sum(counts))
plt.plot(bin_edges[1:],pdf);

plt.show();

#REFER program.txt
# Need for Cumulative Distribution Function (CDF)
# We can visually see what percentage of versicolor flowers have a 
# petal_length of less than 5?
# How to construct a CDF?
# How to read a CDF?

#Plot CDF of petal_length
#print(iris_setosa['petal.length'])
print(np.histogram(student_performance['reading score'], bins=10, 
                                 density = False))
counts, bin_edges = np.histogram(student_performance['reading score'], bins=10, 
                                 density = True)
#print(max(iris_setosa['petal.length']))
#print(min(iris_setosa['petal.length']))
#print("counts:",counts)
#print("Sum:",sum(counts))
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)


counts, bin_edges = np.histogram(student_performance['reading score'], bins=5, 
                                 density = True)
#print (counts)
#print (bin_edges)
pdf = counts/(sum(counts))
plt.plot(bin_edges[1:],pdf);

plt.show();

#REFER program.txt
# Need for Cumulative Distribution Function (CDF)
# We can visually see what percentage of versicolor flowers have a 
# petal_length of less than 5?
# How to construct a CDF?
# How to read a CDF?

#Plot CDF of petal_length
#print(iris_setosa['petal.length'])
print(np.histogram(student_performance['writing score'], bins=10, 
                                 density = False))
counts, bin_edges = np.histogram(student_performance['writing score'], bins=10, 
                                 density = True)
#print(max(iris_setosa['petal.length']))
#print(min(iris_setosa['petal.length']))
#print("counts:",counts)
#print("Sum:",sum(counts))
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)


counts, bin_edges = np.histogram(student_performance['writing score'], bins=5, 
                                 density = True)
#print (counts)
#print (bin_edges)
pdf = counts/(sum(counts))
plt.plot(bin_edges[1:],pdf);

plt.show();

# Plots of CDF of petal_length for various types of flowers.

# Misclassification error if you use petal_length only.

counts, bin_edges = np.histogram(student_performance['math score'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)


# virginica
counts, bin_edges = np.histogram(student_performance['reading score'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)


#versicolor
counts, bin_edges = np.histogram(student_performance['writing score'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)


plt.show();