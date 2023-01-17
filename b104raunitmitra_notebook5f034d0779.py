from matplotlib import pyplot as plt

col=['c' for i in range(8)]

col[4]='red'

plt.bar(['O+', 'A+', 'B+', 'AB+', 'O-', 'A-', 'B-', 'AB-'],[2, 13, 15, 5, 3, 4, 5, 3], color=col)

plt.legend()

plt.xlabel('Blood Groups')

plt.ylabel('No. of Patients')

plt.title('Blood Group Data Set')

plt.show()
marks=[64, 50, 71, 95, 98, 83]

subj=['English', 'Bengali', 'Hindi', 'Maths', 'History' , 'Geography']

colors=['c', 'r', 'b', 'y', 'c', 'g']

plt.pie(marks, labels=subj, colors=colors, startangle=90, shadow=True, explode=[0,0.1,0,0,0,0], autopct='%1.2f%%')

plt.title("Marksheet as Pie Chart")

plt.show()
heights=[72,71,56,45,67,89,54,58,67,77,77,78,77,73,73,172,72,71,56,45,67,

         89,54,58,67,172,77,78,77,73,73,172,12,54,64,75,75,77,88,66,70,12,54,64,75,75,77,88,66,70]

def plot_his(heights):

    start=min(heights)-min(heights)%10

    end=max(heights)+10

    bins=list(range(start,end,5))

    plt.hist(heights,bins,histtype='bar',rwidth=0.5,color='#FF2400')

    plt.xlabel('heights in inches')

    plt.ylabel('No. of Students')

    plt.title("Heights chart")

    plt.show()

print("Abnormal Data")

plot_his(heights)

heights=list(filter(lambda x: not x==172 and not x==12, heights))

print("Correct Data")

plot_his(heights)