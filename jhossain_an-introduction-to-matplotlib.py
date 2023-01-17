import matplotlib.pyplot as plt # plt is the convention, also known as nickname 
# <---Fake Data for Plotting---->

# Median ages 

ages = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]



# Median Microbiologist Salaries by Age

mib_salary = [38496, 42000, 46752, 49320, 53200, 56000, 62316, 64928, 67317, 68748, 73752]



# Median Pharmacist Salaries by Age

pharma_salary = [45372, 48876, 53850, 57287, 63016,65998, 70003, 70000, 71496, 75370, 83640]



# Median Cader Salaries by Age

bcs_salary = [37810, 43515, 46823, 49293, 53437,56373, 62375, 66674, 68745, 68746, 74583]
plt.plot(ages, mib_salary)

plt.show() 
plt.plot(ages, mib_salary)

plt.xlabel('Ages')

plt.ylabel('Median Salary (USD)')

plt.title('Median Salary (USD) by Age')

plt.show() 
# Microbiology

plt.plot(ages, mib_salary)

# Pharmacy

plt.plot(ages, pharma_salary)

plt.xlabel('Ages')

plt.ylabel('Median Salary (USD)')

plt.title('Median Salary (USD) by Age')

plt.show() 
# Microbiology

plt.plot(ages, mib_salary, label="Microbiology")

# Pharmacy

plt.plot(ages, pharma_salary, label= "Pharmacy")

# BCS

plt.plot(ages, bcs_salary, label= "BCS")



plt.xlabel('Ages')

plt.ylabel('Median Salary (USD)')

plt.title('Median Salary (USD) by Age')

plt.legend() 

plt.show() 
# Microbiology

plt.plot(ages, mib_salary, label="Microbiology")

# Pharmacy

plt.plot(ages, pharma_salary, label= "Pharmacy")

# BCS

plt.plot(ages, bcs_salary, label= "BCS")



plt.xlabel('Ages')

plt.ylabel('Median Salary (USD)')

plt.title('Median Salary (USD) by Age')

plt.legend(loc="best") 

plt.show() 
# Microbiology

plt.plot(ages, mib_salary, label="Microbiology")

# Pharmacy

plt.plot(ages, pharma_salary, label= "Pharmacy")

# BCS

plt.plot(ages, bcs_salary, label= "BCS")



plt.xlabel('Ages')

plt.ylabel('Median Salary (USD)')

plt.title('Median Salary (USD) by Age')

plt.legend() 

plt.tight_layout()

plt.show() 
# Microbiology

plt.plot(ages, mib_salary, label="Microbiology",  color="b", linewidth=2, marker='o')

# Pharmacy

plt.plot(ages, pharma_salary, label= "Pharmacy", color="red", linewidth=3, marker='x')

# BCS

plt.plot(ages, bcs_salary, label= "BCS", linewidth=4, linestyle='--')



plt.xlabel('Ages')

plt.ylabel('Median Salary (USD)')

plt.title('Median Salary (USD) by Age')

plt.legend() 

plt.tight_layout()

plt.show() 
# Microbiology

plt.plot(ages, mib_salary, label="Microbiology", color='#444444')

# Pharmacy

plt.plot(ages, pharma_salary, label= "Pharmacy")

# BCS

plt.plot(ages, bcs_salary, label= "BCS", linestyle='--')



plt.xlabel('Ages')

plt.ylabel('Median Salary (USD)')

plt.title('Median Salary (USD) by Age')

plt.legend() 

plt.tight_layout()

plt.show() 
# Microbiology

plt.plot(ages, mib_salary, label="Microbiology",  color="b", linewidth=2, marker='o')

# Pharmacy

plt.plot(ages, pharma_salary, label= "Pharmacy", color="red", linewidth=3, marker='x')

# BCS

plt.plot(ages, bcs_salary, label= "BCS", linewidth=4, linestyle='--')



plt.xlabel('Ages', fontsize=12)

plt.ylabel('Median Salary (USD)', fontsize=12)

plt.title('Median Salary (USD) by Age', fontsize=12)

plt.legend() 

plt.tight_layout()

plt.show() 
plt.figure(figsize=(10,6))

# Microbiology

plt.plot(ages, mib_salary, label="Microbiology",  color="b", linewidth=2, marker='o')

# Pharmacy

plt.plot(ages, pharma_salary, label= "Pharmacy", color="red", linewidth=3, marker='x')

# BCS

plt.plot(ages, bcs_salary, label= "BCS", linewidth=4, linestyle='--')



plt.xlabel('Ages')

plt.ylabel('Median Salary (USD)')

plt.title('Median Salary (USD) by Age')

plt.legend() 

plt.tight_layout()

plt.show() 
# Available styles 

plt.style.available
plt.figure(figsize=(10,6))

# plt.style.use("fivethirtyeight")

# Microbiology

plt.plot(ages, mib_salary, label="Microbiology",  color="b", linewidth=2, marker='o')

# Pharmacy

plt.plot(ages, pharma_salary, label= "Pharmacy", color="red", linewidth=3, marker='x')

# BCS

plt.plot(ages, bcs_salary, label= "BCS", linewidth=4, linestyle='--')



plt.xlabel('Ages')

plt.ylabel('Median Salary (USD)')

plt.title('Median Salary (USD) by Age')

plt.legend() 

plt.tight_layout()

plt.show() 
plt.figure(figsize=(10,6))

# plt.style.use("fivethirtyeight")

# Microbiology

plt.plot(ages, mib_salary, label="Microbiology",  color="b", linewidth=2, marker='o')

# Pharmacy

plt.plot(ages, pharma_salary, label= "Pharmacy", color="red", linewidth=3, marker='x')

# BCS

plt.plot(ages, bcs_salary, label= "BCS", linewidth=4, linestyle='--')



plt.xlabel('Ages')

plt.ylabel('Median Salary (USD)')

plt.title('Median Salary (USD) by Age')

plt.legend() 

plt.tight_layout()

plt.savefig("median_salary.pdf") # .pdf, .png, .jpg, .jepg, .tiff

plt.show() 
fig, ax = plt.subplots(nrows=2, ncols=3)

print(ax)
fig, ax = plt.subplots(2, 3, sharex='col', sharey='row')
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)  

# plt.style.use("fivethirtyeight")

# Microbiology

ax1.plot(ages, mib_salary, label="Microbiology",  color="b", linewidth=2, marker='o')

# Pharmacy

ax1.plot(ages, pharma_salary, label= "Pharmacy", color="red", linewidth=3, marker='x')

# BCS

ax2.plot(ages, bcs_salary, label= "BCS", linewidth=4, linestyle='--')



ax1.set_xlabel('Ages')

ax1.set_ylabel('Median Salary (USD)')

ax1.set_title('Median Salary (USD) by Age')

ax1.legend() 



ax2.set_xlabel('Ages')

ax2.set_ylabel('Median Salary (USD)')

ax2.set_title('Median Salary (USD) by Age')

ax2.legend() 



plt.tight_layout()

plt.savefig("median_salary.pdf") # .pdf, .png, .jpg, .jepg, .tiff

plt.show() 
import random

weight = [random.random() for i in range(20)]
plt.hist(weight) 

plt.xlabel("Weight")

plt.ylabel("Frequency")

plt.title("Distribution of Weight")

plt.show() 
plt.hist(weight, bins=15) 

plt.xlabel("Weight")

plt.ylabel("Frequency")

plt.title("Distribution of Weight")

plt.show() 
# Height in(m)

height = [1.4, 1.2, 1.5, 1.3, 1.6, 1.5]

# Weight in (Kg) 

weight = [60, 15, 85, 74, 77, 65]
plt.scatter(weight, height)

plt.xlabel("Weight")

plt.ylabel("Weight") 

plt.title("Scatter Plot of Height & Weight") 

plt.tight_layout() 

plt.show() 
import random

collection = [random.random() for i in range(10)]

plt.boxplot(collection)

plt.show() 
import random

collectn_1 = [random.random() for i in range(10)]

collectn_2 = [random.random() for i in range(15)]

collectn_3 = [random.random() for i in range(20)]

collectn_4 = [random.random() for i in range(15)]

collections = [collectn_1, collectn_2, collectn_3, collectn_4]

plt.boxplot(collections)

plt.show() 
# <---Fake Data for Plotting---->

# Gender 

gender = ["Female", "Male", "Female", "Male", "Male", "Female"]



# Age group 

age_group = ["Adult", "Child", "Adult", "Adult","Adult","Elderly"]



# Height in(m)

height = [1.4, 1.2, 1.5, 1.3, 1.6, 1.5]



# Weight in (Kg) 

weight = [60, 15, 85, 74, 77, 65]
plt.bar(gender, height)

plt.xlabel("Gender")

plt.ylabel("Height") 

plt.title("Example Bar Graph")

plt.show() 
plt.bar(age_group, height)

plt.xlabel("Age Group")

plt.ylabel("Height") 

plt.title("Example Bar Graph")

plt.show() 
# <---Fake Data for Plotting---->

labels = ["A", "T", "G", "C", "N"]

sizes = [1057, 1184, 2089, 1267, 89]
plt.pie(sizes, labels=labels)

plt.show() 
plt.pie(sizes, labels=labels, autopct='%1.1f%%')

plt.show() 
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']

plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors)

plt.show() 
plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, shadow=True)

plt.show() 
explode = (0, 0, 0.1, 0, 0)   # only "explode" the 3rdslice (i.e. 'Hogs')
plt.figure(figsize=(10,6))

plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, shadow=True, startangle=90, explode=explode)

plt.legend(loc="best")

plt.tight_layout() 

plt.show() 