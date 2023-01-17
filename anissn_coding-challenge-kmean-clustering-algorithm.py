import math as m
class Label_Vector:
    
    def __init__(self, label, size):
        self.label = label
        self.size = size
        self.nums = []
        
    def __repr__(self):
        return self.label + " (" + str(self.size) + ") : " + str(self.nums)
    
    def get_value(self, i):
        return self.nums[i]
    
    def set_value(self, i, value):
        self.nums[i] = value
        
    def set_nums(self, values):
        if len(values) <= self.size:
            self.nums = values
        else:
            print("The set has reached the max size >_<")
    
    def add(self, value):
        if len(self.nums) < self.size:
            self.nums.append(value)
        else:
            print(str(value) + " : The set has reached the max size >_<")
            
    def sum_nums(self, values):
        if len(values) == len(self.nums):
            result = []
            for i in range(len(values)):
                result.append(self.nums[i] + values[i])
            return result
        else:
            return "Vectors sizes don't match >_<"
    
    def mulscalar(self, scalar):
        result = []
        for elmt in self.nums:
            result.append(elmt * scalar)
        return result

    def dotproduct(self, vector):
        if len(vector.nums) == len(self.nums):
            dp = 0
            for i in range(len(vector.nums)):
                dp = dp + self.nums[i]*vector.nums[i]
            return dp
        else:
            return "Vectors sizes don't match >_<"
    
    def distance(self, vector):
        if len(vector.nums) == len(self.nums) and len(self.nums) == self.size:
            dp = 0
            for i in range(len(vector.nums)):
                dp = dp + m.pow(self.nums[i]-vector.nums[i],2)
            return m.sqrt(dp)
        else:
            return "Problem with size >_<"
import csv
def open_csv(file_name, header = True):
    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        if header == True:
            line_count = 0
        else:
            line_count = 1
        vecs = []
        for row in csv_reader:
            if line_count != 0:
                vec_temp = Label_Vector(row[len(row)-1],len(row)-2)
                for i in range(1,len(row)-1):
                    vec_temp.add(float(row[i]))

                vecs.append(vec_temp)
                line_count += 1
            else:
                line_count += 1
        return vecs
iris = open_csv("../input/iris.csv")
def closest_dist(itemA, items): #works only if the items have unique elements
    if itemA != items[0]:
        closest = items[0]
    else:
        closest = items[1]
    
    for elmt in items:
        if itemA.distance(elmt) < itemA.distance(closest) and itemA != elmt:
            closest = elmt
    return closest
import random
class Kmean:
    
    def __init__(self, K, dataset):
        self.K = K
        indexes = random.sample(range(0, len(dataset)-1), K)
        self.centroids = []
        for i in indexes:
            self.centroids.append(dataset[i])
        self.update(dataset)
        self.p_per_clust(dataset)
        
    def __repr__(self):
        return "K : " + str(self.K) + " >> " + str(self.centroids)
    
    def centroid_i(self, item):
        f_index = 0
        for i in range(self.K):
            if item.distance(self.centroids[i]) < item.distance(self.centroids[f_index]):
                f_index = i
        return f_index
    
    def clusters_i(self, items):
        f_list = []
        for elmt in items:
            f_list.append(self.centroid_i(elmt))
        return f_list
    
    def update(self, dataset, new_c = [], count = 0):
        if count != 100 and new_c != self.centroids:
            
            if new_c != []:
                self.centroids = new_c
            clustered = self.clusters_i(dataset)

            new_c = []
            for cn in range(self.K):
                div = 0
                sum = [0]*len(dataset[0].nums)
                
                for i in range(len(dataset)):
                    if clustered[i] == cn:
                        sum = dataset[i].sum_nums(sum)
                        div += 1
                
                for i in range(len(sum)):
                    sum[i] = sum[i] / div
                
                temp_c = Label_Vector("",len(sum))
                temp_c.set_nums(sum)
                new_c.append(temp_c)
            count += 1
            self.update(dataset,new_c,count)
        else:
            print(self)

    def p_per_clust(self, dataset):
        cat = []
        for elmt in dataset:
            if elmt.label not in cat:
                cat.append(elmt.label)
        ppc = []
        clustered = self.clusters_i(dataset)
        tot_cat = [0]*len(cat)
        for cn in range(self.K):
            temp = [0]*len(cat)
            for i in range(len(dataset)):
                if clustered[i] == cn:
                    for i2 in range(len(cat)):
                        if dataset[i].label == cat[i2]:
                            temp[i2] += 1
                            tot_cat[i2] += 1
            ppc.append(temp)
        for cn in range(self.K):
            for i in range(len(cat)):
                ppc[cn][i] = round((ppc[cn][i] / tot_cat[i])*100,1)
        
        print("Categories per cluster : " + str(cat))
        for cn in range(self.K):
            print("Cluster" + str(cn) + " : " + str(ppc[cn]))
            
        self.clusters_names = []
        for cn in range(self.K):
            max_i = 0
            for i in range(len(cat)):
                if ppc[cn][i] > ppc[cn][max_i]:
                    max_i = i
            self.clusters_names.append(cat[max_i])
        return ppc

    def predict_cat(self, value):
        return self.clusters_names[self.centroid_i(value)]
    
    def predict_values(self, values):
        prediction = []
        errors = 0
        for elmt in values:
            prediction.append(self.predict_cat(elmt))
            if self.predict_cat(elmt) != elmt.label:
                errors += 1
        print("Accuracy : " + str(((len(values)-errors)/len(values))*100))
        return prediction
kk = Kmean(3,iris)
import plotly.express as px
x = []
y = []
lbl = []
kmean_lbl = []
clustered = kk.clusters_i(iris)
i=0
for elmt in iris:
    x.append(elmt.nums[0])
    y.append(elmt.nums[1])
    lbl.append(elmt.label)
    kmean_lbl.append(str(clustered[i]))
    i+=1
fig = px.scatter(x=x, y=y, color=kmean_lbl)
fig.show()
fig = px.scatter(x=x, y=y, color=lbl)
fig.show()
kk.p_per_clust(iris)
kk.clusters_names
kk.predict_values(iris)