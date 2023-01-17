!pip install ripser
import matplotlib.pyplot as plt

import numpy as np

from ripser import ripser, Rips

from persim import plot_diagrams

from sklearn.datasets import make_circles, make_blobs, make_moons, load_wine, load_breast_cancer, load_iris

import random
x, y= [.3, .9, .74, .76], [.4, .21, .74, .18]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3,  figsize=(10,2))

ax1.set_xlim([0, 1])

ax1.set_ylim([0, 1])

ax1.scatter([x[0], x[2]], [y[0], y[2]], c='black')

ax1.title.set_text('0-simplexes')

ax1.axis('off')

ax2.set_xlim([0, 1])

ax2.set_ylim([0, 1])

ax2.scatter(x[:2], y[:2], c='black')

ax2.plot([x[0], x[1]],[y[0], y[1]], c='black')

ax2.title.set_text('1-simplex')

ax2.axis('off')

ax3.set_xlim([0, 1])

ax3.set_ylim([0, 1])

ax3.scatter(x[:3], y[:3], c='black')

ax3.plot([x[0],x[1]],[y[0],y[1]], c='black')

ax3.plot([x[1],x[2]],[y[1],y[2]], c='black')

ax3.plot([x[0],x[2]],[y[0],y[2]], c='black')

ax3.fill_between([x[2], x[0], x[1], x[2]], [y[2], y[0], y[1], y[2]], alpha=.5)

ax3.title.set_text('2-simplex')

ax3.axis('off')
x, y= [.3, .9, .74, .76, .23, .44, .16, .58, .5], [.4, .21, .74, .18, .11, .56, .77, .49, .3]

fig, ax = plt.subplots(1, figsize=(4, 4))

ax.set_xlim([0, 1])

ax.set_ylim([0, 1])

ax.scatter(x, y, c='black')

ax.plot([x[0],x[6]],[y[0],y[6]], c='black')

ax.plot([x[0],x[5]],[y[0],y[5]], c='black')

ax.plot([x[6],x[5]],[y[6],y[5]], c='black')

ax.fill_between([x[0], x[6], x[5], x[0]], [y[0], y[6], y[5], y[0]], alpha=.5)

ax.plot([x[4],x[0]],[y[4],y[0]], c='black')

ax.plot([x[7],x[5]],[y[7],y[5]], c='black')

ax.plot([x[7],x[2]],[y[7],y[2]], c='black')

ax.plot([x[7],x[3]],[y[7],y[3]], c='black')

ax.plot([x[2],x[3]],[y[2],y[3]], c='black')

ax.fill_between([x[7], x[3], x[2], x[7]], [y[7], y[3], y[2], y[7]], alpha=.5)

ax.plot([x[1],x[3]],[y[1],y[3]], c='black')

ax.plot([x[0],x[8]],[y[0],y[8]], c='black')

ax.plot([x[4],x[8]],[y[4],y[8]], c='black')

ax.fill_between([x[0], x[8], x[4], x[0]], [y[0], y[8], y[4], y[0]], alpha=.5)

ax.axis('off')

ax.title.set_text('2-simplicial complex')
x, y = [.3, .29, .44, .81, .8], [.21, .35, .6, .56, .45]



def line_ax2(x,y,p1,p2):

    x1, x2 = x[p1], x[p2]

    y1, y2 = y[p1], y[p2]

    ax2.plot([x1,x2],[y1,y2],'r--')



def line_ax3(x,y,p1,p2):

    x1, x2 = x[p1], x[p2]

    y1, y2 = y[p1], y[p2]

    ax3.plot([x1,x2],[y1,y2],'r--')



fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5))

ax1.set_xlim([0, 1])

ax1.set_ylim([0, 1])

ax1.scatter(x, y)

ax1.title.set_text('Radius equal to 0.045 - no touches occur')

ax1.axis('off')

for i in range(len(x)):

    ball_2d = plt.Circle((x[i], y[i]), .045 , color='b', fill=True, alpha=.25)

    ax1.add_artist(ball_2d)



ax2.set_xlim([0, 1])

ax2.set_ylim([0, 1])

ax2.scatter(x, y)

line_ax2(x,y,0,1)

line_ax2(x,y,3,4)

ax2.title.set_text('Radius equal to 0.14 - two touches occur')

ax2.axis('off')

for i in range(len(x)):

    ball_2d = plt.Circle((x[i], y[i]), .14 , color='b', fill=True, alpha=.25)

    ax2.add_artist(ball_2d)



ax3.set_xlim([-.1, 1.1])

ax3.set_ylim([-.1, 1.1])

ax3.scatter(x, y)

line_ax3(x,y,0,1)

line_ax3(x,y,1,2)

#line_ax3(x,y,0,2)

line_ax3(x,y,3,4)

ax3.title.set_text('Radius equal to 0.29 - third touch occurs')

ax3.axis('off')

for i in range(len(x)):

    ball_2d = plt.Circle((x[i], y[i]), 0.29 , color='b', fill=True, alpha=.2)

    ax3.add_artist(ball_2d)
data = np.array([x, y]).T

rips = Rips(maxdim=0)

dgms = rips.fit_transform(data)

#dgms = ripser(data, maxdim=0)['dgms']

print('\nOne can find precise points, where each connected component starts/dies:\n\n', dgms[0])   

plt.title('Persistent Homology Plot')

rips.plot(dgms, show=True)

plt.show() 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))

ax1.set_xlim([0, 1])

ax1.set_ylim([0, .9])

ax1.scatter(x[:4], y[:4])

ax1.scatter(x[4], y[4], c='r')

ax1.plot([x[4],x[3]],[y[4],y[3]],'r--')

ax1.title.set_text('Radius equal to  0.11045361')

ax1.axis('off')

for i in range(len(x)):

    ball_2d = plt.Circle((x[i], y[i]), dgms[0][0][1] , color='b', fill=True, alpha=.25)

    ax1.add_artist(ball_2d)

dgms = ripser(data, maxdim=0, thresh=.14)['dgms']

plt.title('Persistent Homology Plot')

plot_diagrams(dgms, show=True)

plt.show()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))

ax1.set_xlim([0, 1])

ax1.set_ylim([0, .9])

ax1.scatter(x[1:], y[1:])

ax1.scatter(x[0], y[0], c='r')

ax1.plot([x[4],x[3]],[y[4],y[3]],'b-')

ax1.plot([x[0],x[1]],[y[0],y[1]],'r--')

ax1.title.set_text('Radius equal to  0.14035669')

ax1.axis('off')

for i in range(len(x)):

    ball_ax1 = plt.Circle((x[i], y[i]), dgms[0][1][1] , color='b', fill=True, alpha=.25)

    ax1.add_artist(ball_ax1)   

dgms = ripser(data, maxdim=0, thresh=.15)['dgms']

plt.title('Persistent Homology Plot')

plot_diagrams(dgms, show=True)

plt.show()
rips_ax2 = Rips(maxdim=0, thresh=.35)

rips_ax4 = Rips(maxdim=0, thresh=.65)

dgms_ax2 = rips_ax2.fit_transform(data)

dgms_ax4 = rips_ax4.fit_transform(data)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))



ax1.set_xlim([0, 1])

ax1.set_ylim([0, .9])

ax1.scatter(x, y)

ax1.scatter(x[1], y[1], c='r')

ax1.plot([x[4],x[3]],[y[4],y[3]],'b-')

ax1.plot([x[0],x[1]],[y[0],y[1]],'b-')

ax1.plot([x[2],x[1]],[y[2],y[1]],'r--')

ax1.title.set_text('Radius equal to  0.2915476')

ax1.axis('off')

ax3.set_xlim([0, 1])

ax3.set_ylim([0, .9])

ax3.scatter(x, y)

ax3.plot([x[4],x[3]],[y[4],y[3]],'b-')

ax3.plot([x[0],x[1]],[y[0],y[1]],'b-')

ax3.plot([x[2],x[1]],[y[2],y[1]],'b-')

ax3.plot([x[2],x[3]],[y[2],y[3]],'r--')

ax3.title.set_text('Radius equal to  0.37215587')

ax3.scatter(x[3], y[3], c='r')

ax3.axis('off')    

ax2.title.set_text('Persistent Homology Plot')



for i in range(len(x)):

    ball_ax1 = plt.Circle((x[i], y[i]), dgms_ax2[0][2][1] , color='b', fill=True, alpha=.25)

    ball_ax3 = plt.Circle((x[i], y[i]), dgms_ax4[0][3][1] , color='b', fill=True, alpha=.18)

    ax1.add_artist(ball_ax1)

    ax3.add_artist(ball_ax3)



rips.plot(dgms_ax2, show=False, ax=ax2)

rips.plot(dgms_ax4, show=False, ax=ax4)



plt.show()
dgms = ripser(data, maxdim=0)['dgms']

print(dgms[0])
def create_half_moons_with_balls_and_0_homologies(n_samples, noise=.1, radius=np.inf, maxdim=0):

    if radius>.4:   ## i wanted to avoid too large balls on the plot

        radius_homology=radius

        radius=.4

    else:

        radius_homology=radius

        

    global data, labels

    data, labels = make_moons(n_samples, noise=noise, shuffle=True, random_state=42)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))

    title = 'Radius = ' + str(radius_homology)

    ax1.title.set_text(title)

    ax1.set_xlim([np.min(data[:,0][labels==0])-.2, np.max(data[:,0][labels==0])+.2])

    ax1.set_ylim([np.min(data[:,1][labels==0])-.2, np.max(data[:,1][labels==0])+.2])

    ax1.scatter(data[:,0][labels==0], data[:,1][labels==0], c=labels[labels==0], cmap=plt.get_cmap('winter'))

    ax1.axis('off')

    for i in range(len(data)):

        if labels[i]==0:

            ball_2d = plt.Circle((data[i,0], data[i,1]), radius, color='b', fill=False)

            ax1.add_artist(ball_2d)

    plt.title('The persistent homology plot')     

    dgms = ripser(data[labels==0], maxdim=maxdim, thresh=radius_homology)['dgms']

    plot_diagrams(dgms, show=False, ax=ax2)

    plt.show()

    differences = []

    if radius_homology==np.inf:

        for i in range(int(n_samples/2)-2):

            differences.append(np.sort(dgms[0][i+1])-np.sort(dgms[0][i]))

        diff_mean = np.mean(np.array(differences)[:,1])

        diff_std = np.std(np.array(differences)[:,1])

        print('The average distance between persistent homology points is',diff_mean,'. The standard deviation equals ',diff_std)   
create_half_moons_with_balls_and_0_homologies(100, .05, .05)

create_half_moons_with_balls_and_0_homologies(100, .05, .1)

create_half_moons_with_balls_and_0_homologies(100, .05) ##no radius parameter means it's equal to infinity, so we can check on the right plot what is the biggest value of radius once the last death occurs
def create_moons_with_balls_and_0_homologies(n_samples, noise=.1, radius=.05, maxdim=0):

    if radius>.4:

        radius_homology=radius

        radius=.4

    else:

        radius_homology=radius

        

    global data, labels

    data, labels = make_moons(n_samples, noise=noise, shuffle=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))

    ax1.set_xlim([np.min(data[:,0])-.2, np.max(data[:,0])+.2])

    ax1.set_ylim([np.min(data[:,1])-.2, np.max(data[:,1])+.2])

    ax1.scatter(data[:,0], data[:,1], c=labels, cmap=plt.get_cmap('winter'))

    ax1.axis('off')

    for i in range(len(data)):

        if labels[i]==0:

            ball_2d = plt.Circle((data[i,0], data[i,1]), radius, color='b', fill=False)

            ax1.add_artist(ball_2d)

        else:

            ball_2d = plt.Circle((data[i,0], data[i,1]), radius, color='g', fill=False)

            ax1.add_artist(ball_2d)

    plt.title('The persistent homology plot')

    dgms = ripser(data, maxdim=maxdim, thresh=radius_homology)['dgms']

    plot_diagrams(dgms, show=False, ax=ax2)

    last_death = np.round(np.max(np.sort(dgms[0])[:n_samples-1]),2)

    second_last_death = np.round(np.max(np.sort(dgms[0])[:n_samples-2]), 2)

    differences = []

    for i in range(n_samples-2):

        differences.append(np.sort(dgms[0][i+1])-np.sort(dgms[0][i]))

    diff_mean = np.mean(np.array(differences)[:,1])

    print('The value of radius once the last death occurs, is equal to', last_death, ', the second last death occurs for radius', second_last_death, '.\nThe difference between last two deaths is', np.round(last_death-second_last_death, 5), 'once the average difference between each next two points is', np.round(diff_mean, 5))

    plt.show()
create_moons_with_balls_and_0_homologies(100, .06, 1)
x, y = [0, 0, 2, 2], [0, 1, 0, 1]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,3))

ax1.set_xlim([-1, 3])

ax1.set_ylim([-1, 2])

ax1.scatter(x, y)

ax1.scatter(x[0], y[0], c='r')

ax1.scatter(x[3], y[3], c='g')

ax1.plot([x[0],x[1]],[y[0],y[1]],'b--')

ax1.plot([x[3],x[2]],[y[3],y[2]],'b--')

ax1.plot([x[1],x[3]],[y[1],y[3]],'b--')

ax1.title.set_text('red and green points are indirectly connected')

ax1.axis('off')

ax2.set_xlim([-1, 3])

ax2.set_ylim([-1, 2])

ax2.scatter(x, y)

ax2.scatter(x[0], y[0], c='r')

ax2.scatter(x[3], y[3], c='g')

ax2.plot([x[0],x[1]],[y[0],y[1]],'b--')

ax2.plot([x[3],x[2]],[y[3],y[2]],'b--')

ax2.plot([x[1],x[3]],[y[1],y[3]],'b--')

ax2.plot([x[0],x[3]],[y[0],y[3]],'g--')

ax2.title.set_text('red and green points are directly connected')

ax2.axis('off')
x, y = [0, 0, 2, 2], [0, 1, 0, 1]



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))

ax1.set_xlim([-1, 3])

ax1.set_ylim([-1, 2])

ax1.scatter(x, y)

ax1.scatter(x[2], y[2], c='r')

ax1.plot([x[0],x[1]],[y[0],y[1]],'b-')

ax1.plot([x[3],x[2]],[y[3],y[2]],'b--')

ax1.plot([x[1],x[3]],[y[1],y[3]],'b-')

ax1.plot([x[0],x[3]],[y[0],y[3]],'b-')

ax1.fill_between([x[0], x[2], x[1]], [y[0], y[3], y[1]], alpha=.5)

ax1.title.set_text("blue points are 1-connected")

ax1.axis('off')

ax2.set_xlim([-1, 3])

ax2.set_ylim([-1, 2])

ax2.plot([x[0]-.1,x[1]-.1],[y[0],y[1]],'b-')

ax2.plot([x[0],x[2]],[y[1]+.1,y[3]+.1],'b-')

ax2.plot([x[0]+.1,x[2]+.1],[y[0]-.1,y[1]-.1],'b-')

ax2.plot([x[1],x[3]],[y[1],y[3]],'b--')

ax2.plot([x[0],x[3]],[y[0],y[3]],'b--')

ax2.plot([x[0],x[1]],[y[0],y[1]],'b--')

ax2.plot([x[1],x[3]],[y[1],y[3]],'b--')

ax2.plot([x[0],x[3]],[y[0],y[3]],'b--')

ax2.fill_between([x[0], x[2], x[1]], [y[0], y[3], y[1]], alpha=.5)

ax2.title.set_text("traingle with its 'boundaries' ")

ax2.axis('off')
data, labels = make_circles(n_samples=65, shuffle=True, noise=.12)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))

ax1.set_xlim([-1.5, 1.5])

ax1.set_ylim([-1.5, 1.5])

ax1.scatter(data[:,0], data[:,1], c='b')

ax1.axis('off')

for i in range(len(data)):

    ball_2d = plt.Circle((data[i,0], data[i,1]), .145  , color='b', fill=True, alpha=.18)

    ax1.add_artist(ball_2d)

    

dgms = ripser(data, maxdim=1)['dgms']

plot_diagrams(dgms, show=True)

plt.show()
def create_moons_with_balls_and_0_1_homologies(n_samples, noise=.1, radius=np.inf, maxdim=1):

    if radius>.4:

        radius_homology=radius

        radius=.4

    else:

        radius_homology=radius

        

    global data, labels

    data, labels = make_moons(n_samples, noise=noise, shuffle=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))

    ax1.set_xlim([np.min(data[:,0])-.2, np.max(data[:,0])+.2])

    ax1.set_ylim([np.min(data[:,1])-.2, np.max(data[:,1])+.2])

    ax1.scatter(data[:,0], data[:,1], c=labels, cmap=plt.get_cmap('winter'))

    ax1.axis('off')

    for i in range(len(data)):

        if labels[i]==0:

            ball_2d = plt.Circle((data[i,0], data[i,1]), radius, color='b', fill=False)

            ax1.add_artist(ball_2d)

        else:

            ball_2d = plt.Circle((data[i,0], data[i,1]), radius, color='g', fill=False)

            ax1.add_artist(ball_2d)

    plt.title('The persistent homology plot')

    dgms = ripser(data, maxdim=maxdim, thresh=radius_homology)['dgms']

    plot_diagrams(dgms, show=False, ax=ax2)

    last_death = np.round(np.max(np.sort(dgms[0])[:n_samples-1]),2)

    second_last_death = np.round(np.max(np.sort(dgms[0])[:n_samples-2]), 2)

    differences = []

    for i in range(n_samples-2):

        differences.append(np.sort(dgms[0][i+1])-np.sort(dgms[0][i]))

    diff_mean = np.mean(np.array(differences)[:,1])

    print('The value of radius once the last death occurs, is equal to', last_death, ', the second last death occurs for radius', second_last_death, '.\nThe difference between last two deaths is', np.round(last_death-second_last_death, 5), 'once the average difference between each next two points is', np.round(diff_mean, 5))

    plt.show()
create_moons_with_balls_and_0_1_homologies(150, .06)
def circles_with_balls_and_0_1_homologies(n_samples, noise=.1, radius=np.inf, maxdim=1, factor=0.4):

    if radius>.4:

        radius_homology=radius

        radius=.4

    else:

        radius_homology=radius

        

    global data, labels

    data, labels = make_circles(n_samples, noise=noise, shuffle=True, factor=factor)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))

    ax1.title.set_text("make_circle dataset")

    ax1.set_xlim([np.min(data[:,0])-.2, np.max(data[:,0])+.2])

    ax1.set_ylim([np.min(data[:,1])-.2, np.max(data[:,1])+.2])

    ax1.scatter(data[:,0], data[:,1], c=labels, cmap=plt.get_cmap('winter'))

    ax1.axis('off')

    for i in range(len(data)):

        if labels[i]==0:

            ball_2d = plt.Circle((data[i,0], data[i,1]), radius, color='b', fill=False, alpha=.5)

            #ax1.add_artist(ball_2d)

        else:

            ball_2d = plt.Circle((data[i,0], data[i,1]), radius, color='g', fill=False, alpha=.5)

            #ax1.add_artist(ball_2d)

    plt.title('The persistent homology plot')

    dgms = ripser(data, maxdim=maxdim, thresh=radius_homology)['dgms']

    plot_diagrams(dgms, show=False, ax=ax2)

    last_death = np.round(np.max(np.sort(dgms[0])[:n_samples-1]),2)

    second_last_death = np.round(np.max(np.sort(dgms[0])[:n_samples-2]), 2)

    differences = []

    for i in range(n_samples-2):

        differences.append(np.sort(dgms[0][i+1])-np.sort(dgms[0][i]))

    diff_mean = np.mean(np.array(differences)[:,1])

    print('The value of radius once the last death occurs, is equal to', last_death, ', the second last death occurs for radius', second_last_death, '.\nThe difference between last two deaths is', np.round(last_death-second_last_death, 5), 'once the average difference between each next two points is', np.round(diff_mean, 5))

    plt.show()
def blobs_with_balls_and_0_1_homologies(n_samples, radius=np.inf, maxdim=1):

    if radius>.4:

        radius_homology=radius

        radius=.4

    else:

        radius_homology=radius

        

    global data, labels

    data, labels = make_blobs(n_samples=n_samples, cluster_std=[.5, .5, .5], centers=3)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))

    ax1.title.set_text("make_blobs dataset")

    ax1.set_xlim([np.min(data[:,0])-.2, np.max(data[:,0])+.2])

    ax1.set_ylim([np.min(data[:,1])-.2, np.max(data[:,1])+.2])

    ax1.scatter(data[:,0], data[:,1], c=labels, cmap=plt.get_cmap('winter'))

    ax1.axis('off')

    for i in range(len(data)):

        if labels[i]==0:

            ball_2d = plt.Circle((data[i,0], data[i,1]), radius, color='b', fill=False, alpha=.5)

            #ax1.add_artist(ball_2d)

        else:

            ball_2d = plt.Circle((data[i,0], data[i,1]), radius, color='g', fill=False, alpha=.5)

            #ax1.add_artist(ball_2d)

    plt.title('The persistent homology plot')

    dgms = ripser(data, maxdim=maxdim, thresh=radius_homology)['dgms']

    plot_diagrams(dgms, show=False, ax=ax2)

    last_death = np.round(np.max(np.sort(dgms[0])[:n_samples-1]),2)

    second_last_death = np.round(np.max(np.sort(dgms[0])[:n_samples-2]), 2)

    differences = []

    for i in range(n_samples-2):

        differences.append(np.sort(dgms[0][i+1])-np.sort(dgms[0][i]))

    diff_mean = np.mean(np.array(differences)[:,1])

    print('On the below persistent diagram, 0-homologies should suggest 3 clusters in dataset')

    plt.show()
def wine_with_balls_and_0_1_homologies(radius=np.inf, maxdim=1):

    if radius>.4:

        radius_homology=radius

        radius=.4

    else:

        radius_homology=radius

        

    global data, labels

    wine = load_wine()

    data, labels = wine.data, wine.target

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))

    ax1.set_xlim([np.min(data[:,0])-.2, np.max(data[:,0])+.2])

    ax1.set_ylim([np.min(data[:,1])-.2, np.max(data[:,1])+.2])

    ax1.scatter(data[:,0], data[:,1], c=labels)

    ax1.axis('off')

    ax1.title.set_text("load_wine dataset")

    for i in range(len(data)):

        if labels[i]==0:

            ball_2d = plt.Circle((data[i,0], data[i,1]), radius, color='b', fill=False, alpha=.5)

            #ax1.add_artist(ball_2d)

        elif labels[i]==1:

            ball_2d = plt.Circle((data[i,0], data[i,1]), radius, color='g', fill=False, alpha=.5)

            #ax1.add_artist(ball_2d)

        else:

            ball_2d = plt.Circle((data[i,0], data[i,1]), radius, color='r', fill=False, alpha=.5)

            #ax1.add_artist(ball_2d)

    plt.title('The persistent homology plot')

    dgms = ripser(data, maxdim=maxdim, thresh=radius_homology)['dgms']

    plot_diagrams(dgms, show=False, ax=ax2)

    differences = []

    for i in range(len(dgms[0])-2):

        differences.append(np.sort(dgms[0][i+1])-np.sort(dgms[0][i]))

    diff_mean = np.mean(np.array(differences)[:,1])

    plt.show()
def cancer_with_balls_and_0_1_homologies(radius=np.inf, maxdim=1):

    if radius>.4:

        radius_homology=radius

        radius=.4

    else:

        radius_homology=radius

        

    global data, labels

    cancer = load_breast_cancer()

    data, labels = cancer.data, cancer.target

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))

    ax1.title.set_text("load_breast_cancer dataset")

    ax1.set_xlim([np.min(data[:,0])-.2, np.max(data[:,0])+.2])

    ax1.set_ylim([np.min(data[:,1])-.2, np.max(data[:,1])+.2])

    ax1.scatter(data[:,0], data[:,1], c=labels)

    ax1.axis('off')

    for i in range(len(data)):

        if labels[i]==0:

            ball_2d = plt.Circle((data[i,0], data[i,1]), radius, color='b', fill=False, alpha=.5)

            #ax1.add_artist(ball_2d)

        elif labels[i]==1:

            ball_2d = plt.Circle((data[i,0], data[i,1]), radius, color='g', fill=False, alpha=.5)

            #ax1.add_artist(ball_2d)

        else:

            ball_2d = plt.Circle((data[i,0], data[i,1]), radius, color='r', fill=False, alpha=.5)

            #ax1.add_artist(ball_2d)

    plt.title('The persistent homology plot')

    dgms = ripser(data, maxdim=maxdim, thresh=radius_homology)['dgms']

    plot_diagrams(dgms, show=False, ax=ax2)

    differences = []

    for i in range(len(dgms[0])-2):

        differences.append(np.sort(dgms[0][i+1])-np.sort(dgms[0][i]))

    diff_mean = np.mean(np.array(differences)[:,1])

    plt.show()
def iris_with_balls_and_0_1_homologies(radius=np.inf, maxdim=1):

    if radius>.4:

        radius_homology=radius

        radius=.4

    else:

        radius_homology=radius

        

    global data, labels

    iris = load_iris()

    data, labels = iris.data, iris.target

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))

    ax1.title.set_text("load_breast_cancer dataset")

    ax1.set_xlim([np.min(data[:,0])-.2, np.max(data[:,0])+.2])

    ax1.set_ylim([np.min(data[:,1])-.2, np.max(data[:,1])+.2])

    ax1.scatter(data[:,0], data[:,1], c=labels)

    ax1.axis('off')

    for i in range(len(data)):

        if labels[i]==0:

            ball_2d = plt.Circle((data[i,0], data[i,1]), radius, color='b', fill=False, alpha=.5)

            #ax1.add_artist(ball_2d)

        elif labels[i]==1:

            ball_2d = plt.Circle((data[i,0], data[i,1]), radius, color='g', fill=False, alpha=.5)

            #ax1.add_artist(ball_2d)

        else:

            ball_2d = plt.Circle((data[i,0], data[i,1]), radius, color='r', fill=False, alpha=.5)

            #ax1.add_artist(ball_2d)

    plt.title('The persistent homology plot')

    dgms = ripser(data, maxdim=maxdim, thresh=radius_homology)['dgms']

    plot_diagrams(dgms, show=False, ax=ax2)

    differences = []

    for i in range(len(dgms[0])-2):

        differences.append(np.sort(dgms[0][i+1])-np.sort(dgms[0][i]))

    diff_mean = np.mean(np.array(differences)[:,1])

    plt.show()
circles_with_balls_and_0_1_homologies(100, 0.06)

blobs_with_balls_and_0_1_homologies(100)

wine_with_balls_and_0_1_homologies()

cancer_with_balls_and_0_1_homologies(800)

iris_with_balls_and_0_1_homologies()