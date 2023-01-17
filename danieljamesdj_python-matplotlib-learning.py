import matplotlib.pyplot as plt

import numpy as np
plt.plot([1, 3, 6], [2, 4, 5])

plt.show()
plt.plot([1, 3, 6], [2, 4, 5])

plt.title('Simple graph')

plt.xlabel('x')

plt.ylabel('y')

plt.legend(['Line'])

plt.show()
fig, ax = plt.subplots(2, 3) # This will create 6 axes, with 3 in each row.
fig, ax = plt.subplots(2, 3)

ax[0][2].plot([1, 3, 6], [2, 4, 5])

ax[0][2].set_title('Simple Graph')

ax[0][2].set_xlabel('x')

ax[0][2].set_ylabel('y')

ax[0][2].legend(['Line'])

plt.show()
# We will plot the line graph of the runs scored by Sachin and Kohli in ODIs.



_, axis = plt.subplots()



x1 = np.arange(1,24) # Number of years after starting the ODI career of Sachin

y1 = [239,656,1360,1679,2768,3212,4823,5834,7728,8571,9899,10803,11544,12685,13497,13909,14537,15962,16422,17394,17598,18111,18426] # Runs scored by Sachin in ODI at the end of each year

x2 = np.arange(1,13) # Number of years after starting the ODI career of Kohli

y2 = [159,484,1479,2860,3886,5154,6208,6831,7570,9030,10232,11520] # Runs scored by Kohli in ODI at the end of each year



axis.plot(x1, y1, label='Sachin')

axis.plot(x2, y2, label='Virat')

axis.set_title('Sachin v/s Kohli run comparison in ODI')

axis.set_xlabel('Number of years')

axis.set_ylabel('Runs')

axis.legend()

plt.show()
# We will plot the average scored by players against different teams.



_, axis = plt.subplots()



x_labels = ['Australia', 'Pakistan', 'England', 'New Zealand', 'Sri Lanka']

y1 = [31.94, 28.89, 36.65, 73.67, 50.55] # Averages by Gambhir against teams

y2 = [21.69, 34.55, 37.33, 52.59, 34.67] # Averages by Sehwag against teams

y3 = [25, 32, 34, 20, 25] # Averages by Kohli against teams

y4 = [25, 32, 34, 20, 25] # Averages by Yuvraj against teams

y5 = [25, 32, 34, 20, 25] # Averages by Dhoni against teams

x = np.arange(len(x_labels)) # Get the values of x-axis

width = 0.18 # Width of one bar graph



axis.bar(x - 2 * width, y1, width, label='Gambhir')

axis.bar(x - width, y2, width, label='Sehwag')

axis.bar(x, y3, width, label='Kohli')

axis.bar(x + width, y4, width, label='Yuvraj')

axis.bar(x + 2 * width, y5, width, label='Dhoni')

axis.set_ylabel('Average')

axis.set_title('Average by players against different teams')

axis.set_xticks(x)

axis.set_xticklabels(x_labels)

axis.legend()

plt.show()
# We will plot the frequency of run ranges of Sehwag in each ODI innings where he scored at least a half century.



_, axis = plt.subplots()



scores = [31,4,34,3,15,96,30,5,0,20,10,219,0,26,20,0,38,15,73,39,5,35,175,28,110,12,99,19,10,11,9,46,42,13,47,10,44,4,146,6,38,30,11,40,13,40,125,3,54,77,6,5,116,42,91,69,68,1,85,60,42,49,119,78,2,59,89,17,14,11,33,6,10,43,25,8,52,45,21,30,48,114,2,46,12,19,11,18,0,65,17,9,10,1,8,9,95,11,97,12,22,73,4,26,15,26,7,67,5,27,30,77,1,35,22,19,48,39,38,20,37,21,6,75,12,45,15,5,38,21,48,6,32,2,14,21,5,29,2,74,108,45,70,0,53,10,17,1,0,4,17,5,81,1,16,37,0,20,26,13,26,79,12,3,23,32,90,35,5,130,39,0,0,31,8,25,43,37,63,82,33,1,66,3,21,23,24,36,4,6,4,112,45,23,7,108,0,12,18,52,4,114,1,28,25,13,59,126,48,45,39,46,0,16,12,71,32,0,21,31,42,82,51,5,29,34,55,4,55,33,5,4,100,0,27,33,12,0,2,4,11,2,58,19,1]

# Individual score of Sehwag in every innings

ranges = np.arange(50,250,12.5)



axis.hist(scores, ranges, histtype='bar', rwidth=0.8)

axis.set_xlabel('Run ranges')

axis.set_ylabel('Frequency')

axis.set_title('Frequency of Sehwag\'s 50+ scores in ODI')

plt.show()
# We will plot the runs scored by Sehwag in every ODI innings.



_, axis = plt.subplots()



scores = [31,4,34,3,15,96,30,5,0,20,10,219,0,26,20,0,38,15,73,39,5,35,175,28,110,12,99,19,10,11,9,46,42,13,47,10,44,4,146,6,38,30,11,40,13,40,125,3,54,77,6,5,116,42,91,69,68,1,85,60,42,49,119,78,2,59,89,17,14,11,33,6,10,43,25,8,52,45,21,30,48,114,2,46,12,19,11,18,0,65,17,9,10,1,8,9,95,11,97,12,22,73,4,26,15,26,7,67,5,27,30,77,1,35,22,19,48,39,38,20,37,21,6,75,12,45,15,5,38,21,48,6,32,2,14,21,5,29,2,74,108,45,70,0,53,10,17,1,0,4,17,5,81,1,16,37,0,20,26,13,26,79,12,3,23,32,90,35,5,130,39,0,0,31,8,25,43,37,63,82,33,1,66,3,21,23,24,36,4,6,4,112,45,23,7,108,0,12,18,52,4,114,1,28,25,13,59,126,48,45,39,46,0,16,12,71,32,0,21,31,42,82,51,5,29,34,55,4,55,33,5,4,100,0,27,33,12,0,2,4,11,2,58,19,1]

y = np.arange(len(scores))



axis.scatter(y, scores, label='Runs in n-th innings')

axis.set_xlabel('Innings')

axis.set_ylabel('Runs')

axis.set_title('Runs scored by Sehwag in every ODI innings')

axis.legend()

plt.show()
# We will plot the runs scored by different players in ODI every year.



_, axis = plt.subplots()



years = np.arange(1999,2020)

# Runs scored by different players every year

gambhir = [0,0,0,0,113,0,181,115,634,1119,848,670,720,685,153,0,0,0,0,0,0]

sehwag = [1,19,439,1130,871,671,1017,608,475,893,810,446,645,217,31,0,0,0,0,0,0]

kohli = [0,0,0,0,0,0,0,0,0,159,325,995,1381,1026,1268,1054,623,739,1460,1202,1288]

yuvraj = [0,260,238,659,600,841,839,849,1287,893,783,349,453,2,276,0,0,0,372,0,0]

dhoni = [0,0,0,0,0,19,895,821,1103,1097,1198,600,764,524,753,418,640,278,788,275,600]



axis.stackplot(years, gambhir, sehwag, kohli, yuvraj, dhoni, labels=["Gambhir","Sehwag", "Kohli", "Yuvraj", "Dhoni"])

axis.set_xlabel('Years')

axis.set_ylabel('Runs')

axis.set_title('Runs scored by players in ODI yearwise')

axis.legend()

plt.show()
# We will plot the number of matches captained by different captains of Indian cricket team.



_, axis = plt.subplots()



captainedInnings = [37,74,174,73,146,79,200,80]

playerNames = ['Sunil Gavaskar','Kapil Dev','Mohammad Azharuddin','Sachin Tendulkar','Sourav Ganguly','Rahul Dravid','Mahendra Singh Dhoni','Virat Kohli']



axis.pie(captainedInnings, labels = playerNames, explode=(0,0,0,0,0,0,0,0.1))

axis.set_title('Indian Captains (captained at least 20 matches)')

plt.show()