import numpy as np

import itertools

#import pandas as pd

class Solution:

    def __init__(self):

        pass

    def threeSum(self, nums): #-> list[list[int]]:

        if(len(nums)>=3):

            P = np.array(list(itertools.combinations(nums, 3)))

            P.sort(axis=1)

            P = P[P.sum(axis = 1) == 0].tolist()

            P.sort()

            list1 = []

            for i,j in enumerate(P[0:len(P)-1]):

                if(j == P[i+1]):

                    pass

                else:

                    list1.append(j)

            try:

                list1.append(P[-1])

            except:

                pass

            

            return list1

        else:

            return []
L =[-2,-7,-11,-8,9,-7,-8,-15,10,4,3,9,8,11,1,12,-6,-14,-2,-1,

    -7,-13,-11,-15,11,-2,7,-4,12,7,-3,-5,7,-7,3,2,1,10,2,-12,-1,

    12,12,-8,9,-9,11,10,14,-6,-6,-8,-3,-2,14,-15,3,-2,-4,1,-9,8,11,5,-14,-1,

    14,-6,-14,2,-2,-8,-9,-13,0,7,-7,-4,2,-8,-2,11,-9,2,-13,-10,2,5,4,13,

    13,2,-1,10,14,-8,-14,14,2,10]

target = 0



class two_sum():

    

    def __init__(self):

        pass

    

    def two_sum_list(self, list_nums):

        

        list_two_sum = []

        list_nums.sort()

        i = 0 

        j =  len(list_nums) - 1

        

        while(i < j):

            if((list_nums[i] + list_nums[j]) > target ):

                j = j-1

                #pass

            elif((list_nums[i] + list_nums[j]) < target):

                i = i+1

                #pass

            else:

                list_two_sum.append((list_nums[i], list_nums[j]))

                i = i+1

                j = j-1

       

    ## Removing Duplicates

        

        list_two_sum.sort()

        list1 = []

        for i,j in enumerate(list_two_sum[0:len(list_two_sum)-1]):

            if(j == list_two_sum[i+1]):

                pass

            else:

                list1.append(j)

        try:

            list1.append(list_two_sum[-1])

        except:

            pass

        

        return list1
two_sum().two_sum_list(L)
target = 0



class ThreeSum():

    

    def __init__(self):

        pass

    

    def three_sum_list(self, list_nums):

        

        list_sum = []

        list_nums.sort()

        

        for k in range(len(list_nums)):

            i = k+1 

            j =  len(list_nums) - 1



            while(i < j):

                if((list_nums[k] + list_nums[i] + list_nums[j]) > target ):

                    j = j-1

                    #pass

                elif((list_nums[k] + list_nums[i] + list_nums[j]) < target):

                    i = i+1

                    #pass

                else:

                    list_sum.append((list_nums[k], list_nums[i], list_nums[j]))

                    i = i+1

                    j = j-1

       

    ## Removing Duplicates

        

        list_sum.sort()

        list1 = []

        for i,j in enumerate(list_sum[0:len(list_sum)-1]):

            if(j == list_sum[i+1]):

                pass

            else:

                list1.append(j)

        try:

            list1.append(list_sum[-1])

        except:

            pass

        

        return list1
ThreeSum().three_sum_list(L)
L =  [1,0,-1,0,-2,2] #[-3,-1,0,2,4,5] #

target = 0



class FourSum():

    

    def __init__(self):

        pass

    

    def four_sum_list(self, list_nums):

        if(len(list_nums)>=4):

            list_sum = []

            list_nums.sort()

            for l in range(len(list_nums)):

                for k in range(len(list_nums[l+1:])):

                    i = (l+1) + (k+1) 

                    j =  len(list_nums) - 1



                    while(i < j):

                        S = list_nums[l] + list_nums[l+1+k] + list_nums[i] + list_nums[j]

                        if(S > target ):

                            #print(S)

                            j = j-1

                            #pass

                        elif(S < target):

                            #print(S)

                            i = i+1

                            #pass

                        else:

                            #print(S)

                            list_sum.append((list_nums[l] , list_nums[l+1+k], list_nums[i], list_nums[j]))

                            i = i+1

                            j = j-1



        ## Removing Duplicates



            list_sum.sort()

            list1 = []

            for i,j in enumerate(list_sum[0:len(list_sum)-1]):

                if(j == list_sum[i+1]):

                    pass

                else:

                    list1.append(j)

            try:

                list1.append(list_sum[-1])

            except:

                pass



            return list1

        else:

            return []
FourSum().four_sum_list(L)