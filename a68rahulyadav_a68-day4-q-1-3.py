"""Q1.Store blood groups of 50 different atients and show the no of 

patients having O- blood grouQ1. Store blood groups of 50 different 

atients and show the no of patients having O- blood group"""



from matplotlib import pyplot as plt

blood_grp=['O+', 'A+', 'B+', 'AB+', 'O-', 'A-', 'B-', 'AB-']

patients=[5, 10, 12, 5, 3, 4, 5, 6]

colors = ['b','b','b','b','g','b','b','b']

plt.bar(blood_grp,patients,color=colors)

plt.legend()

plt.xlabel('Blood Groups')

plt.ylabel('No. of Patients')

plt.title('Blood Group Data Set')

plt.show()
"""Q2. Store data of marks acquired by a certain student and show 

them in form of a piechart and slice out the subject having least 

marks"""



import matplotlib.pyplot as plt

subjects=['English','Bengali','Hindi','Maths','History','Geography']

marks = [87,89,93,92,98,95]

plt.pie(marks,labels=subjects,startangle=90,shadow=True,

        explode=(0.2,0,0,0,0,0),autopct='%1.2f%%')

plt.show()
"""Q3.Store data of heights of 50 students with 4 mistakes and plot 

them in a graph and segregate normal data from abnormal one"""



heights=[72,71,56,45,67,89,54,58,67,77,77,78,77,73,73,172,72,71,56,

         45,67,89,54,58,67,172,77,78,77,73,73,172,12,54,64,75,75,77,

         88,66,70,12,54,64,75,75,77,88,66,70]

def plot_his(heights):

    start=min(heights)-min(heights)%10

    end=max(heights)+10

    bins=list(range(start,end,5))

    plt.hist(heights,bins,histtype='bar',rwidth=0.5,color='c')

    plt.xlabel('height of students (inches)')

    plt.ylabel('No.of Students')

    plt.show()

plot_his(heights)

heights=list(filter(lambda x: not x==172 and not x==12, heights))

plot_his(heights)