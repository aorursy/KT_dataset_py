import matplotlib.pyplot as plt 
  
#blood groups
groups=['O+','A+','B+','AB+','O-','A-','B-','AB-']

# frequencies 
freq = [15,3,7,9,7,5,1,3]
  
# labels for bars 
tick_label = ['O+','A+','B+','AB+','O-','A-','B-','AB-'] 
  
# plotting a bar chart 
plt.bar(groups, freq, tick_label = tick_label, width = 0.8, color = ['red', 'green']) 
  
# naming the x-axis 
plt.xlabel('Blood Groups') 
# naming the y-axis 
plt.ylabel('Frequency') 
# plot title 
plt.title('Blood Group Graph') 
  
# function to show the plot 
plt.show() 
import matplotlib.pyplot as plt 

# defining labels 
subjects = ['English', 'Bengali', 'Hindi', 'Maths', 'History', 'Geography'] 

# portion covered by each label 
slices = [30, 70, 80, 60, 50, 82] 

# color for each label 
colors = ['red', 'yellow', 'green', 'blue','orange','chocolate'] 

# plotting the pie chart 
plt.pie(slices, labels = subjects, colors=colors, autopct = '%0.1f%%',  explode = (0.1, 0, 0, 0, 0, 0)) 

plt.axis('equal') 

# showing the plot 
plt.show()
import numpy as np
import matplotlib.pyplot as plt

# defining heights
heights=np.array([172, 61, 75, 61, 69, 61, 12, 63, 60, 79, 68, 63, 79, 68, 69, 70, 68, 68, 65, 73, 63, 71, 62, 66
, 69, 66, 77, 62, 77, 63, 70, 66, 172, 74, 62 ,79, 69, 77, 64, 72, 77, 76, 72, 78, 78, 60, 73, 76,
 70, 12])

# plot the histogram
plt.hist(heights)
plt.show()
print('Original Heights:',heights)

# change the heights
heights[heights==12]=72
heights[heights==172]=72

# plot the new histogram
plt.hist(heights)
plt.show()
print('New Heights:',heights)