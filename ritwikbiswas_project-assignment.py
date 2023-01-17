#local file path stuff (skip if need be)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        #print(os.path.join(dirname, filename))

        pass
%matplotlib inline



import collections

import sys

import matplotlib

import matplotlib.pyplot as plt

import numpy as np

import pandas

from matplotlib import colors

from matplotlib.ticker import PercentFormatter




class Classroom:

    def __init__(self, P, S, R, K):

        self.P, self.S, self.K, self.R, self.Q = P, S, K, R, sys.maxsize

        self.projects = collections.OrderedDict()

        self.students = collections.OrderedDict()

    

    def generate_data(self):

        ''' Generates mock project and random student preferences based on P,S,K values '''

        for i in range(0,self.P): #project generation

            proj_name = ''

            if i > 25:

                temp = i

                while temp >= 0:

                    proj_name = chr(ord('a')+(temp%26)) + proj_name

                    temp = int(temp/26) - 1

                print(proj_name)

            else:

                proj_name = chr(ord('a')+i)

            self.projects[proj_name] = []

        

        for i in range(0,self.S): #student generation

            sname = 's' + str(i)

            import random

            #assignments = random.sample(range(self.P), self.K)

            assignments = random.choices(range(self.P),weights=(40,20,10,5,5,5,5,5,1,1,1,1,1),k=self.K) #weighted

            items = list(self.projects.items())

            preferences = []

            for n in assignments:

                preferences.append(items[n][0])

            self.students[sname] = {'Preferences': preferences, "Q": sys.maxsize} #Q is set to max int initially

            

    def pretty_print(self,only_projects=False):

        ''' Clean print of data with or without student info'''

        print("\nProjects and Assignments")

        for project in self.projects:

            print("project: %s %s"% (project,self.projects[project]))

        if not only_projects:

            print("\nStudents and Preferences")

            for student in self.students:

                print("%s, pref: %s, Q: %s"% (student,self.students[student]['Preferences'],self.students[student]['Q']))

                

    def graph_Q_dist(self):

        ''' Graph the Q distribution across all Qs'''

        import matplotlib.pyplot as plt

        fig = plt.figure()

        ax = fig.add_axes([0,0,1,1])

        q_dict = {}

        q_scores = []

        q_values = []

        for student in self.students:

            q_score = self.students[student]['Q']

            if q_score not in q_dict:

                q_dict[q_score] = 1

            else:

                q_dict[q_score] += 1

        for i in range(1,self.K):

            q_values.append(i)

            if i in q_dict:

                q_scores.append(q_dict[i])

            else:

                q_scores.append(0)

        

        ax.bar(q_values,q_scores)

        plt.show()

        

    def unassigned_students(self):

        ''' Find the number of unassigned students'''

        number = 0

        for student in self.students:

#             print(self.students[student]['Q'])

            if self.students[student]['Q'] == sys.maxsize:

                number+= 1

#         print("unass: %s"%(number))

        return number

        

    def compute_avgQ(self):

        ''' Calculate avg Q for all students'''

        sum = 0.0

        unassigned = 0

        assigned_sum = 0.0

        for student in self.students:

            sum += self.students[student]['Q']

            if self.students[student]['Q'] == sys.maxsize:

                unassigned +=1

            else:

                assigned_sum += self.students[student]['Q']

        avgQ = sum/float(self.S)

        avgQ_assigned = assigned_sum/float(self.S)

        print("Avg Q: %.5f, Unassigned: %s"%(avgQ,unassigned))

        

        return avgQ

    

    def projects_below_R(self):

        ''' Checks which projects have below R assignments and returns them'''

        project_list = []

        for project in self.projects:

            if len(self.projects[project]) < self.R:

                project_list.append(project)

        return project_list

    

    def Q_score(self, student, project):

        ''' Returns student Q score for a given student/project combo (worst Q is maxint) '''

        student_prefs = self.students[student]['Preferences']

        for i in range (0,len(student_prefs)):

            if student_prefs[i] == project:

                return i+1

        return sys.maxsize

    

    def solution_1(self):

        ''' 

        First approach for project assignment

        Iterate through each project that still needs students

        Assign a student to a project as long as no better assignment exists

        Once reached R threshold per project, find any students yet to be assigned and find best option for them else random assignment

        '''

        import time

        start_time = time.time()



        projects_to_assign = self.projects_below_R()

        count = 0

        while len(projects_to_assign)>0: #while there is a project that needs assignment

#             if count > 4:

#                 break

            projects_to_assign_ref = projects_to_assign

            for project in projects_to_assign: #iterate through those projects

                if project not in projects_to_assign_ref:

                    continue

                best_student_candidate = "None"

                best_project_candidate = project

                minQ = sys.maxsize

                for student in self.students: #find best student for that project

                    if self.students[student]['Q'] != sys.maxsize: #skip if student is already assigned

                        continue

                    qscore = self.Q_score(student,project)

                    if qscore <= minQ:

                        best_student_candidate = student

                        minQ = qscore

                        

                #make sure lower Q can't be achieved by assigning student to another project

                for project2 in projects_to_assign_ref:

                    if project2 == project:

                        continue

                    secondary_minQ = self.Q_score(best_student_candidate, project2)

                    if secondary_minQ < minQ and project2:

                        best_project_candidate = project2

                        minQ = secondary_minQ

                        

                #do official assignment of stud -> proj here

                self.projects[best_project_candidate].append(best_student_candidate)

                self.students[best_student_candidate]['Q'] = self.Q_score(best_student_candidate,best_project_candidate)

                #print("Best student for proj %s is %s with Qscore %s"%(best_project_candidate, best_student_candidate, minQ))

                projects_to_assign_ref = self.projects_below_R()

            projects_to_assign = self.projects_below_R()



            count+= 1

        

                

            

        #Assign remaining unassigned students to projects 

        students_to_assign = self.unassigned_students()

        while(students_to_assign>0):

            for project in self.projects:

                best_student_candidate = "None"

                minQ = sys.maxsize

                for student in self.students: #find best student for that project

                    if self.students[student]['Q'] != sys.maxsize: #skip if student is already assigned

                        continue

                    qscore = self.Q_score(student,project)

#                     print("qscore: %s"%(qscore))

#                     print("minq: %s"%(minQ))

#                     print(student)

                    if qscore <= minQ:

                        best_student_candidate = student

                        minQ = qscore 

#                         print(best_student_candidate)

                if self.unassigned_students() == 0:

                    break

                self.projects[project].append(best_student_candidate)

#                 print(self.projects[project])

                score_to_give = self.Q_score(best_student_candidate,project)

                if self.Q_score(best_student_candidate,project) == sys.maxsize:

                    score_to_give = 6

                

                self.students[best_student_candidate]['Q'] = score_to_give

                students_to_assign -=1

                







        

        print("alg1(P%s,S%s,R%s,K%s) execution: %0.5fs" % (self.P,self.S,self.R,self.K,time.time() - start_time))

        return self.compute_avgQ(),self.unassigned_students()
classroom = Classroom(P=13,S=70,R=5,K=7)

classroom.generate_data()

classroom.pretty_print()

q,errors = classroom.solution_1()

classroom.pretty_print()
classroom.graph_Q_dist()
%%capture

test_iterations = 1000

n_bins = 20

q_values = []



for i in range(0,test_iterations):

    c = Classroom(P=13,S=70,R=5,K=7)

    c.generate_data()

    q,errors = c.solution_1()

    q_values.append(q)

fig, axs = plt.subplots(1,sharey=True, tight_layout=True)

axs.hist(q_values, bins=n_bins)