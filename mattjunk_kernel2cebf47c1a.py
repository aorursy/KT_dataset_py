!ls ../input
#ที่ 7:00 น. ใช้ความชื้นอุณหภูมิ กับรถโดยใช้รถหาร 2 ให้เป็นเวลาที่ 7:00 นั้นๆ ใช้รถบวกกันด้วย

import numpy as np

import matplotlib.pyplot as plt

data= np.genfromtxt("../input/-7.00.csv", skip_header=1,skip_footer=5,delimiter= ',')

car= np.genfromtxt("../input/car_data.csv", skip_header=2,delimiter= ',')

car_train = []

car_findk = []

car_test = []

PM_ntrain = []

PM_nfindk = []

PM_ntest = []

humid_train = []

humid_trans_train = []

humid_trans_findk = []

temp_train = []

temp_trans_train = []

temp_trans_findk = []

PM_trans_train = []

humid_findk = []

temp_findk = []

PM_findk = []

humid_test = []

temp_test = []

PM_test = []

PM_train = []

PM_trans_findk = []

PM_trans_test = []

car_trans_train = []

car_trans_findk = []

index = 0

for new in car:

    index+=1

    a = 0

    for b in data:

        a+=1

        if b[6]!=0 and str(b[6])!='nan' and str(b[3])!='nan' and str(b[5])!='nan' and str(new[1])!='nan' and index==a and str(new[2])!='nan':

            if a>=350 and a<=700 :

                humid_trans_train.append(b[3])

                temp_trans_train.append(b[5])

                PM_trans_train.append(b[6])

                PM_train.append(b[6])

                car_trans_train.append(new[1]+new[2])

            elif a>700 and a<=750:

                humid_trans_findk.append(b[3])

                temp_trans_findk.append(b[5])

                PM_trans_findk.append(b[6])

                PM_findk.append(b[6])

                car_trans_findk.append(new[1]+new[2])  

            else :

                humid_test.append(b[3])

                temp_test.append(b[5])

                PM_trans_test.append(b[6])

                PM_test.append(b[6])

                car_test.append(new[1]+new[2]) 

def getKeysBValue(dictOfElements, valueToFind):

    listOfKeys = list()

    listOfItems = dictOfElements.items()

    for item  in listOfItems:

        if item[1] == valueToFind:

            listOfKeys.append(item[0])

    return  listOfKeys

PM_trans_train.pop(0)

PM_ntrain = PM_trans_train

PM_trans_findk.pop(0)

PM_nfindk = PM_trans_findk

PM_trans_test.pop(0)

PM_ntest = PM_trans_test

humid_trans_train.pop(0)

humid_train = humid_trans_train

humid_trans_findk.pop(0)

humid_findk = humid_trans_findk

temp_trans_train.pop(0)

temp_train = temp_trans_train

temp_trans_findk.pop(0)

temp_findk = temp_trans_findk

car_trans_train.pop(0)

car_train = car_trans_train

car_trans_findk.pop(0)

car_findk = car_trans_findk

PM_train.pop(len(PM_train)-1)

PM_findk.pop(len(PM_findk)-1)

PM_test.pop(len(PM_test)-1)

#PM_train = PM_train.pop()
PM_train_dict = dict()

PM_findk_dict = dict()

all_k_all_d = []

l = 0

for k in PM_findk:

    l+=1

    PM_findk_dict[l] = k

l = 0

for k in PM_ntrain:

    l+=1

    PM_train_dict[l] = k
c = 0

for d in humid_findk:

    distance_dict = dict()

    all_k_error_each_d = []

    index2 = 0

    c+=1

    e = 0

    for f in temp_findk:

        e+=1

        aa = 0

        for bb in car_findk:

            aa+=1

            ccc = 0

            for bbb in PM_findk:

                ccc+=1

                if c==e and e==aa and aa==ccc:

                    g = 0

                    for h in humid_train:

                        g+=1

                        i = 0

                        for j in temp_train:

                            i+=1

                            cc = 0

                            for ggg in PM_train:

                                cc+=1

                                sss = 0

                                for dd in car_train:

                                    sss+=1

                                    if g==i and i==cc and cc==sss:

                                        distance = (((d-h)**2)+((f-j)**2)+(((bb-dd))**2)+((bbb-ggg)**2))**(1/2)

                                        distance_dict[i] = distance

                                        index2+=1

                                        print(index2)

    sorted_distance_tuple = sorted(distance_dict.items(), key = lambda kv:(kv[1], kv[0]))

    k_error_dict = dict()

    for k in range(10,40):

        n = 0

        sigma = 0

        sigma_weight = 0

        for m in sorted_distance_tuple:

            n+=1

            if n<=k:

                sigma+=float(PM_train_dict[m[0]])*float(sorted_distance_tuple[k-n][1])

                sigma_weight+=m[1]

                #print(sigma)

                #print(sigma_weight)

        #print(sigma)

        #print(sigma_weight)

        predict = float(sigma)/float(sigma_weight)

        real = PM_findk_dict[c]

        error = (((predict-real)**2)**(1/2)/real)*100

        k_error_dict[k] = error

        all_k_error_each_d.append(k_error_dict)

    all_k_all_d.append(all_k_error_each_d)

    #index2+=1

    #print(index2)
average_error_all_k = dict()

error_plot = []

k_plot = []

#stepXYZ

l = 10

for k in range(10,40):

    x = 0

    z = 0

    for i in all_k_all_d:

        if l==k:

            for j in i:

                if str(j[k])!='nan' and j[k]!=0:

                    x=x+j[k]

                    z+=1

    l+=1

    if z==0:

        z = 1

    y = x/z

#END STEP

    average_error_all_k[k] = y

    error_plot.append(y)

    k_plot.append(k)

plt.bar(k_plot,error_plot)

plt.show()
error = []

for i in average_error_all_k:

    error.append(average_error_all_k[i])

print(getKeysBValue(average_error_all_k, min(error)))