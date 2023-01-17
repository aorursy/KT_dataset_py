import numpy as np

import random 

random.seed()
class genetic(object):

  #generate population

  def gen_population(self,size,low,high):

    #print("Generating Population")

    y=np.zeros([size,size])

    for i in range(0,size):

      for j in range(0,size):

        list_of_route=[]

      for k in range(0,size):

        random_num=random.randint(low,high)

        while(random_num in list_of_route):

          random_num=random.randint(low,high)

        list_of_route.append(random_num)

      y[i]=np.asarray(list_of_route)

    return y



  #fitness

  def fitness(self,samples):

    #print("Fitness")

    sum=0

    total_distance=[]

    for i in range(0,samples.shape[0]):

      for j in range(1,samples.shape[1]):

        sum+=distances[int(samples[i][j-1]),int(samples[i][j])] 

      total_distance.append(sum)

      sum=0

    return total_distance



  #selection

  def selection(self,fit):

    #print("Selection")

    best=[]

    new_fit=fit.copy()

    new_fit.sort()

    size_slice=int(size/2)

    best=new_fit[0:size_slice]

    indexes=[]

    for i in range(len(best)):

      indexes.append(fit.index(best[i]))

    best_sample=samples[indexes].tolist()

    return best_sample,indexes



  #crossover

  def crossover(self,best_samples,indexes):

    #print("Crossover")

    new_offspring=np.zeros([2,4])

    for i in range(len(best_samples)):

      for j in range(len(best_samples[0])):

        if j<(len(best_samples[1]))/2:

          new_offspring[i][j]=best_samples[i][j]

        else:

          if i==0:

            k=0

            while k<len(best_samples[1]):

              if best_samples[i+1][k] not in new_offspring[i]:

                new_offspring[i][j]=best_samples[i+1][k]

                k=5

              k+=1

          else:

            k=0

            while k<len(best_samples[1]):

              if best_samples[i-1][k] not in new_offspring[i]:

                new_offspring[i][j]=best_samples[i-1][k]

                k=5

              k+=1

    return new_offspring



  #mutation

  def mutation(self,crossed):

    #print("Mutation")

    for i in range(crossed.shape[0]):

      random.seed()

      x1=random.randint(0,3)

      x2=random.randint(0,3)

      crossed[i][x1],crossed[i][x2]=crossed[i][x2],crossed[i][x1]

    return crossed
cities=[]

number_of_cities=int(input("Enter number of Cities: "))

for i in range(0,number_of_cities):

  cities.append(str(input("Enter name of  city {}: ".format(i+1))))

city=np.arange(number_of_cities)



distances=np.zeros([number_of_cities,number_of_cities],dtype='int16')

for i in range(0,number_of_cities):

  print("\nDistances from {}: ".format(cities[i]))

  for j in range(0,number_of_cities):

    if i==j:

        x=0

    else:

        x=int(input("Enter the distance from city {} to city {} : ".format(cities[i],cities[j])))

    distances[i][j]=x
k=1

low=0

high=number_of_cities-1

size=number_of_cities

best_distance=10000

best_route=[]

min_distance=[]

torun=True
gene=genetic()

samples=gene.gen_population(size,low,high)

while(torun):

  #print("Iteration : ",k)

  fit=gene.fitness(samples)

  for find_min in fit:

    if find_min<best_distance:

      best_distance=find_min

  min_distance.append(best_distance)

 

  if(best_distance>=min(min_distance)):

    torun=0

  best_sample,indexes=gene.selection(fit)

  best_route=[]

  best_route.append(best_sample[0])

  crossed=gene.crossover(best_sample,indexes)

  new_off=gene.mutation(crossed)

  size_of_sample=len(samples)

  for io in range(int(size_of_sample/2)):

    samples[io]=best_sample[io]

  for io2 in range((len(new_off))):

    io+=1

    samples[io]=new_off[io2]

  k+=1





for i in range(0,len(best_route[0])):

  print(cities[int(best_route[0][i])],end='')

  j=i+1

  if(j!=len(best_route[0])+1):

    print(" --> ",end='') 

print(cities[int(best_route[0][0])])

final_distance=best_distance+distances[int(best_route[0][0])][int(best_route[0][i])]

print("\nTotal Distance to travel: ",final_distance)


