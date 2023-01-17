import numpy as np 

import matplotlib.pyplot as plt

import cv2
class CA_Grid:



    def __init__(self, height=200, width=400, initial_number_of_black_cell=1):

        self.height = height

        self.width = width

        self.initial_number_of_black_cell = initial_number_of_black_cell

        self.grid = None





    def __initialize(self,height, width, initial_number_of_black_cell):

        self.height = height

        self.width = width

        self.grid = None

        self.initial_number_of_black_cell = initial_number_of_black_cell



    def get_grid(self):



        if self.initial_number_of_black_cell==1:

            self.__single_black_cell_grid()

        else:

            self.__multiple_black_cell_grid()



        return self.grid



    def __single_black_cell_grid(self):



        """

        This function creates matrix in heightXwidth dimensions and 

        assigns 1 to the middle cell in the top row of the matrix.



        :return:

        """

        self.grid = np.zeros((self.height, self.width), dtype=np.int32)

        self.grid[0, int(self.width / 2)] = 1









    def __multiple_black_cell_grid(self):



        """

        This function assigns a value of 1 to the desired 

        number of cells of the top row of the heightXwitdth matrix. 

        It ensures that the middle cell is 1.

        :return:

        """







        #Calling the function that assigns the value of the middle cell of the top row to 1.

        self.__single_black_cell_grid()



        """remove 1 from the self.initial_number_of_black_cell variable 

        because the value has been assigned to the middle cell"""

        n=self.initial_number_of_black_cell-1

        for i in range(n):

            random_col = np.random.randint(0, self.width)

            self.grid[0, random_col] = 1



class Elementary_CA(CA_Grid):



    def __init__(self,grid_apparence="normal",**kwargs):

        super().__init__(**kwargs)



        self.grid_apparence=grid_apparence





        self.transform_vector=np.array([4,2,1])

    

        self.rule=None



        self.rule_binary=None



    def set_grid_parameters(self,

                            height,

                            width,

                            initial_number_of_black_cell=1,

                            grid_apparence="normal"):

        self.height = height

        self.width = width

        self.initial_number_of_black_cell = initial_number_of_black_cell

        

        self.grid = None

        self.grid_apparence=grid_apparence



    def __get_rule_binary(self):

        self.rule_binary = np.array([int(b) for b in np.binary_repr(self.rule, 8)], dtype=np.int8)



    def generate(self, rule):



        self.rule=rule

        self.get_grid()

        self.__get_rule_binary()



        for i in range(self.height-1):

            self.grid[i+1,:]=self.step(self.grid[i,:])



        self.grid[self.grid==1]=255



        if self.grid_apparence=="wolfram":

            self.grid=cv2.bitwise_not(self.grid)

        

        return self.grid



    def generate_all_ca(self):

        all_ca=list()

        for i in range(256):

            self.generate(i)

            all_ca.append(self.grid)



        return all_ca





    def __get_neighborhood_matrix(self, center):

        #vector that holds the neighbors on the left by shifting the row vector to the right

        left=np.roll(center, 1)





        #vector that holds the neighbors on the rights by shifting the row vector to the left

        right=np.roll(center, -1)

        

        neighborhood_matrix=np.vstack((left, center, right)).astype(np.int8)



        return neighborhood_matrix



    def step(self, row):

        neighborhood_matrix=self.__get_neighborhood_matrix(center=row)







        #u=self.transform_vector.reshape((3,1))

        #rmts=np.sum(neighborhood_matrix*u, axis=0)



        rmts=self.transform_vector.dot(neighborhood_matrix)



        return self.rule_binary[7-rmts].astype(np.int8)
class Demonstrate_CA:

    def __init__(self):

        print("Demonstrate_CA object created")

    

    def show_rule(self, rule, step):

        step=step

        elementary_CA=Elementary_CA(height=step, width=step*2, grid_apparence="wolfram")

        

        rule=rule

        generated_image=elementary_CA.generate(rule=rule)



        plt.figure(figsize=(15,15))

        plt.imshow(generated_image, cmap="gray")

        plt.xticks([])

        plt.yticks([])

        plt.title("Demonstration of Rule {} for {} Steps".format(rule, step))

        plt.show()

    

    def show_rules_between_0_and_127(self):

        step=30

        elementary_CA=Elementary_CA(height=step, width=step*2, grid_apparence="wolfram")

        all_ca_patterns=elementary_CA.generate_all_ca()

        

        fig,axarr=plt.subplots(nrows=32, ncols=4, figsize=(16, 80))

        axarr=axarr.flatten()

        for index, pattern_image in enumerate(all_ca_patterns[:128]):

            axarr[index].imshow(pattern_image, cmap="gray")

            axarr[index].set_xticks([])

            axarr[index].set_yticks([])

            axarr[index].set_title("Rule {}".format(index))

        #plt.suptitle("Demonstration of Rules Between 0 and 127")

        plt.show()

    

    def show_rules_between_128_and_255(self):

        step=30

        elementary_CA=Elementary_CA(height=step, width=step*2, grid_apparence="wolfram")

        all_ca_patterns=elementary_CA.generate_all_ca()

        

        fig,axarr=plt.subplots(nrows=32, ncols=4, figsize=(16, 80))

        axarr=axarr.flatten()

        for index, pattern_image in enumerate(all_ca_patterns[128:]):

            axarr[index].imshow(pattern_image, cmap="gray")

            axarr[index].set_xticks([])

            axarr[index].set_yticks([])

            axarr[index].set_title("Rule {}".format(index+128))

        #plt.suptitle("Demonstration of Rules Between 128 and 255")

        plt.show()
demonstrate_ca=Demonstrate_CA()
rules=[60, 73,105, 30, 90, 110]

step=100

for rule in rules:

    print("Demontration of rule {} for {} step".format(rule, step))

    demonstrate_ca.show_rule(rule=rule, step=step)
print("Rule Demonstration Between 0 and 127")

demonstrate_ca.show_rules_between_0_and_127()

print("Rule Demontration Between 128 and 255")

demonstrate_ca.show_rules_between_128_and_255()
class Totalistic_CA_Grid:



    def __init__(self, height=200, width=400, initial_number_of_black_cell=1):

        self.height = height

        self.width = width

        self.initial_number_of_black_cell = initial_number_of_black_cell

        self.grid = None





    def __initialize(self,height, width, initial_number_of_black_cell):

        self.height = height

        self.width = width

        self.grid = None

        self.initial_number_of_black_cell = initial_number_of_black_cell



    def get_grid(self):



        if self.initial_number_of_black_cell==1:

            self.__single_black_cell_grid()

        else:

            self.__multiple_black_cell_grid()



        return self.grid



    def __single_black_cell_grid(self):



        """

        This function creates matrix in heightXwidth dimensions and 

        assigns 1 to the middle cell in the top row of the matrix.



        :return:

        """

        self.grid = np.zeros((self.height, self.width), dtype=np.int32)

        self.grid[0, int(self.width / 2)] = 2









    def __multiple_black_cell_grid(self):



        """

        This function assigns a value of 1 to the desired 

        number of cells of the top row of the heightXwitdth matrix. 

        It ensures that the middle cell is 1.

        :return:

        """







        #Calling the function that assigns the value of the middle cell of the top row to 1.

        self.__single_black_cell_grid()



        """remove 1 from the self.initial_number_of_black_cell variable 

        because the value has been assigned to the middle cell"""

        n=self.initial_number_of_black_cell-1

        for i in range(n):

            random_col = np.random.randint(0, self.width)

            self.grid[0, random_col] = 2



class Totalistic_CA(Totalistic_CA_Grid):



    def __init__(self,grid_apparence="normal",**kwargs):

        super().__init__(**kwargs)



        self.grid_apparence=grid_apparence



        self.rule=None



        self.rule_tenary=None



    def set_grid_parameters(self,

                            height,

                            width,

                            initial_number_of_black_cell=1,

                            grid_apparence="normal"):

        self.height = height

        self.width = width

        self.initial_number_of_black_cell = initial_number_of_black_cell

        

        self.grid = None

        self.grid_apparence=grid_apparence



    def __get_rule_tenary(self):

        length=7

        if self.rule==0:

            padding=length

        else:

            padding=length-len(np.base_repr(self.rule,base=3))

            

        self.rule_tenary = np.array([int(b) for b in np.base_repr(

            number=self.rule, 

            base=3,

            padding=padding)], dtype=np.int8)

        



    def generate(self, rule):



        self.rule=rule

        self.get_grid()

        self.__get_rule_tenary()



        for i in range(self.height-1):

            self.grid[i+1,:]=self.step(self.grid[i,:])



        if self.grid_apparence=='normal':

            self.grid[self.grid==2]=255

            self.grid[self.grid==1]=128

            self.grid[self.grid==0]=0

        

        if self.grid_apparence=='wolfram':

            self.grid[self.grid==1]=128

            self.grid[self.grid==0]=255

            self.grid[self.grid==2]=0

        

        return self.grid





    def __get_neighborhood_matrix(self, center):

        #vector that holds the neighbors on the left by shifting the row vector to the right

        left=np.roll(center, 1)





        #vector that holds the neighbors on the rights by shifting the row vector to the left

        right=np.roll(center, -1)

        

        neighborhood_matrix=np.vstack((left, center, right)).astype(np.int8)



        return neighborhood_matrix



    def step(self, row):

        neighborhood_matrix=self.__get_neighborhood_matrix(center=row)



        rmts=np.sum(neighborhood_matrix, axis=0)

        #print("rmts",rmts)



        return self.rule_tenary[6-rmts].astype(np.int8)
totalistic_rules=[224, 2049, 966, 1041, 993, 777]

for totalistic_rule in totalistic_rules:

    totalistic_ca=Totalistic_CA(grid_apparence="wolfram")

    ca_pattern=totalistic_ca.generate(totalistic_rule)

    plt.figure(figsize=(12,6))

    plt.imshow(ca_pattern,cmap="gray")

    plt.xticks([])

    plt.yticks([])

    plt.title("Totalistic CA Pattern for Code:{}".format(totalistic_rule))

    plt.show()
totalistic_rules=[2049, 966, 1041, 993, 777,224]

for totalistic_rule in totalistic_rules:

    totalistic_ca=Totalistic_CA(grid_apparence="wolfram", initial_number_of_black_cell=4)

    ca_pattern=totalistic_ca.generate(totalistic_rule)

    plt.figure(figsize=(12,6))

    plt.imshow(ca_pattern,cmap="gray")

    plt.xticks([])

    plt.yticks([])

    plt.title("Totalistic CA Pattern for Code:{}".format(totalistic_rule))

    plt.show()