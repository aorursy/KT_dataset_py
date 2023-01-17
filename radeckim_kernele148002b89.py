import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from pylab import rcParams

import six



import os
#static variables

bar_width = 0.2

table_width = 1

table_hegiht = 2.5



n_groups = 3

index = np.arange(n_groups)



headers_cols2 = ['Academic', 'Professional Services', 'Total']

headers_rows2 = ['Female (Female %)', 'Male (Male %)', 'Total']



headers3 = ['Academic', 'Professional Services', '%Total FT', '%Total PT']

xtra_headers3 = ['Female FT (Female %FT)', 'Female PT (Female %PT)', 'Male FT (Male %FT)', 'Male PT (Male %PT)', \

                 'Female FT Total (%)', 'Male FT Total (%)', 'Female PT Total (%)', 'Male PT Total (%)']



green = ['#CEFFCE', '#84CF96', '#009A31', '#228B22']

yellow = ['#FFFF66', '#FFCC00', '#FF9900', '#FF0000']

others = ['#003f5c', '#bc5090', '#ffa600']



#static params

rcParams['figure.figsize'] = 12, 5

rcParams['font.size'] = 14   
#utility functions



def calc_perc(array, total):

    

    return [str("%.2f" % (round(x/y, 3) * 100))  if y != 0 else 0 for x, y in zip(array, total)]



def calc_perc_float(array, total):

    

    return [float(str("%.2f" % (round(x/y, 3) * 100)))   if y != 0 else 0 for x, y in zip(array, total)]



def num_and_perc_format(nums, pers):

    

    return [str(num) + ' (' + str(per) + '%)'  if per != None else '0' for num, per in zip(nums, pers)]
def graph_bar_table2(fem_tot, male_tot, fem_per, male_per,total, headers_cols, headers_rows, title, yaxis = 'Number of staff'):

       

    plt.figure(figsize = (12, 5))

    ax = plt.subplot()    

    ax.set_axisbelow(False)    

    ax.set_frame_on(False)

    ax.grid(alpha = 0.2)

    ax.set_ylabel('% of staff', fontsize = 12, labelpad = 15, fontweight='bold')

    ax.yaxis.set_tick_params(length = 0)

                    

    ax.bar([x  + bar_width * 2.5 for x in index], 

            fem_per,

            bar_width,             

            color = yellow[0],

            edgecolor = 'black',

            label = 'Female')

    

    ax.bar([x  + bar_width * 2.5  for x in index], 

            male_per, 

            bar_width, 

            bottom = fem_per,

            color = green[0],

            edgecolor = 'black',

            label = 'Male')

           

    plt.xticks([], [])

    plt.yticks(fontsize = 16)

    plt.xlim(0, 3)

    

    fem = [str(num) + ' (' + str(per) + '%)' for num, per in zip(fem_tot, fem_per)]

    male = [str(num) + ' (' + str(per) + '%)' for num, per in zip(male_tot, male_per)]

    

    tot = fem + male

    

    table = plt.table(cellText   = [fem, male, total],

                      colLabels  = headers_cols,

                      loc        = 'bottom',

                      rowLabels = headers_rows,

                      cellLoc  = 'center',

                      rowColours = [yellow[0], green[0], 'lightgrey'],

                      zorder = 100,

                      bbox = [0, -0.3, 1, 0.3]

             ) 

    

    table.scale(table_width, table_hegiht)

    

    if(title == 'Table 21'):

        

            table = plt.table(cellText   = [fem, male, total],

                      colLabels  = headers_cols,

                      loc        = 'bottom',

                      rowLabels = headers_rows,

                      cellLoc  = 'center',

                      rowColours = [yellow[0], green[0], 'lightgrey'],

                      zorder = 100

             ) 



    plt.subplots_adjust(left=0.2, bottom=0.2)

    

    plt.tight_layout()

    plt.show()
##Table 2



fem_total2 = [21, 6, 27]

male_total2 = [24, 1, 25]

fem_per2 = [47, 86, 52]

male_per2 = [53, 14, 48]

total2 = [45, 7, 52]

title2 = 'Table 2'



graph_bar_table2(fem_total2, male_total2, fem_per2, male_per2, total2, headers_cols2, headers_rows2, title2)
def graph_bar_table3_1(fem_ft, fem_ft_per, fem_pt, fem_pt_per, male_ft, male_ft_per, male_pt, male_pt_per, total_pt_fem, total_pt_male, total_ft_fem, total_ft_male):

       

        

        

    ax = plt.subplot()    

    ax.set_axisbelow(False)    

    ax.set_frame_on(False)

    ax.grid(alpha = 0.2)

    ax.set_ylabel('% of staff', fontsize = 14, labelpad = 15, fontweight='bold')

    ax.yaxis.set_tick_params(length = 0)

                    

    ax.bar([x  + bar_width * 2 for x in index], 

            fem_ft_per,

            bar_width,             

            color = yellow[0],

            edgecolor = 'black',

            linewidth = 0.5,

            label = 'Female FT')

    

    ax.bar([x  + bar_width * 2 for x in index], 

            fem_pt_per,

            bar_width,

            bottom = fem_ft_per,                        

            color = yellow[1],

            edgecolor = 'black',

            linewidth = 0.5,

            label = 'Female PT')

    

    ax.bar([x  + bar_width * 3 for x in index], 

            male_ft_per,

            bar_width,             

            color = green[0],

            edgecolor = 'black',

            linewidth = 0.5,

            label = 'Male PT')

    

    ax.bar([x  + bar_width * 3 for x in index], 

            male_pt_per,

            bar_width,

            bottom = male_ft_per,                        

            color = green[1],

            edgecolor = 'black',

            linewidth = 0.5,

            label = 'Male FT')  

           

    plt.xticks([], [])

    plt.yticks(fontsize = 10)

    plt.xlim(0, 3)        

    

    fem_ft = [str(num) + ' (' + str(per) + '%)' for num, per in zip(fem_ft, fem_ft_per)]

    fem_pt = [str(num) + ' (' + str(per) + '%)' for num, per in zip(fem_pt, fem_pt_per)]

    male_ft = [str(num) + ' (' + str(per) + '%)' for num, per in zip(male_ft, male_ft_per)]

    male_pt = [str(num) + ' (' + str(per) + '%)' for num, per in zip(male_pt, male_pt_per)]

      

    table = plt.table(cellText   = [fem_ft, fem_pt, male_ft, male_pt, total_ft_fem, total_ft_male, total_pt_fem, total_pt_male],

                      colLabels  = headers3,

                      loc        = 'bottom',

                      rowLabels = xtra_headers3,

                      cellLoc  = 'center',

                      alpha = 0.5,

                      edges = 'closed',

                      rowColours = [yellow[0], yellow[1], green[0], green[1]] + ['lightgrey'] * 4,

                      zorder = 100,

                      bbox = [0, -0.7, 1, 0.7]

             ) 

              

    table.scale(table_width, table_hegiht)

    

    plt.subplots_adjust(left=0.2, bottom=0.2)

    

    plt.tight_layout()

    plt.show()
## Table 3.1



fem_ft = [13, 2, 15]

fem_ft_per = [62, 33, 55]

fem_pt = [8, 4, 12]

fem_pt_per = [38, 67, 45]

male_ft = [14, 0, 14]

male_ft_per = [58, 0, 56]

male_pt = [10, 1, 11]

male_pt_per = [42, 100, 44]

total_ft_fem = ['48%', '100%', None]

total_ft_male = ['52%', '0%', None]

total_pt_fem = ['45%', '80%', None]

total_pt_male = ['48%', '20%', None]



graph_bar_table3_1(fem_ft, fem_ft_per, fem_pt, fem_pt_per, male_ft, male_ft_per, male_pt, male_pt_per, total_pt_fem, total_pt_male, total_ft_fem, total_ft_male)

#total_pt_fem, total_pt_male, total_ft_fem, total_ft_male
#Table 6



headers_cols6 = ['UG', 'PGT', 'PGR']



fem_total6 = [232, 63, 40]

fem_per6 = [78, 66, 70]

male_total6 = [64, 32, 17]

male_per6 = [22, 34, 30]

total_6 = [296, 95, 57]



title_6 = 'Number and percentage of students by gender and level of study, headcount'

headers_cols_6 = ['UG', 'PGT', 'PGR']

headers_rows_6 = ['Female', 'Male', 'Total']



graph_bar_table2(fem_total6, male_total6, fem_per6, male_per6, total_6, headers_cols_6, headers_rows_6, title_6)
def graph_time12(female_int1, male_int1, total_int1, female_int2, male_int2, total_int2, female_int3, male_int3, total_int3, headers, rows, title, bar_width, offset_id, total):

    

    size = len(female_int1)

    

    fem_all = [calc_perc_float(female_int1, total_int1), calc_perc_float(female_int2, total_int2), calc_perc_float(female_int3, total_int3)]

    male_all = [calc_perc_float(male_int1,total_int1), calc_perc_float(male_int2,total_int2), calc_perc_float(male_int3,total_int3)]

    

    offset = 9

    yoffset = 0    

      

    if(offset_id == 0): 

        

        offset = 18

        plt.figure(figsize=(14, 6.25))

        

    elif(offset_id == 1):

        

        plt.figure(figsize=(10, 4))

        yoffset = -9

        

    elif(offset_id == 3):

        

        fem_all = [[68, 75], [71, 67], [68, 50]]

        male_all = [[32, 25], [29, 33], [32, 50]]

        plt.figure(figsize=(8, 6.6))

        offset = 6

        yoffset = 350

        

    elif(offset_id == 2):

        

        fem_all = [[73, 88], [57, 90], [86, 80]]

        male_all = [[27, 12], [43, 10], [14, 20]]

        plt.figure(figsize=(8, 6.6))

        offset = 6

        yoffset = 350



    fem_all_per = [num_and_perc_format(female_int1, fem_all[0]), \

               num_and_perc_format(female_int2, fem_all[1]), \

               num_and_perc_format(female_int3, fem_all[2])]

    

    male_all_per = [num_and_perc_format(male_int1, male_all[0]), \

                num_and_perc_format(male_int2, male_all[1]), \

                num_and_perc_format(male_int3, male_all[2])]

        

    total_all = [total_int1, total_int1, total_int1]



    size = len(female_int1)    

    ax = plt.subplot()    

    ax.yaxis.set_tick_params(length = 0)   

    ax.set_axisbelow(False)    

    ax.set_frame_on(False)    

    ax.yaxis.grid(alpha = 0.2)

    ax.set_ylabel('% of students', fontsize = 14, labelpad = 15, fontweight='bold')   

    ax.set_ylim(0, 110)

    

    plt.xticks(range(size), [])

    plt.xlim(0, size)    

    

    for i in range(3):

    

        ax.bar([x + bar_width * (i + 2) - bar_width * offset * 0.05 for x in range(size)], 

                fem_all[i],

                bar_width,             

                color = yellow[i],

                edgecolor = 'black',

                linewidth = 0.5,

                label = 'Female' + str(2015 + i) + '- ' + str(2016 + i))     

    

    for i in range(3):

    

        ax.bar([x + bar_width * (i + 2) - bar_width * offset * 0.05 for x in range(size)], 

                male_all[i],

                bar_width,

                bottom = fem_all[i],

                color = green[i],

                edgecolor = 'black',

                linewidth = 0.5,

                label = 'Male' + str(2015 + i) + '- ' + str(2016 + i)) 

        

    perc = [sum(fem_all_per, []) , sum(male_all_per, [])]

    

        

    female_int1_per = calc_perc(female_int1, total_int1)

    male_int1_per = calc_perc(male_int1, total_int1)

    

    female_int2_per = calc_perc(female_int2, total_int2)

    male_int2_per = calc_perc(male_int2, total_int2)



    female_int3_per = calc_perc(female_int3, total_int3)

    male_int3_per = calc_perc(male_int3, total_int3)



    female_int1 = num_and_perc_format(female_int1, female_int1_per)

    male_int1 = num_and_perc_format(male_int1, male_int1_per)



    female_int2 = num_and_perc_format(female_int2, female_int2_per)

    male_int2 = num_and_perc_format(male_int2, male_int2_per)



    female_int3 = num_and_perc_format(female_int3, female_int3_per)

    male_int3 = num_and_perc_format(male_int3, male_int3_per)



    if(total == True):  



        

        table = plt.table(cellText   = [female_int1, female_int2, female_int3, \

                                        male_int1,  male_int2,  male_int3, \

                                        total_int1, total_int2, total_int3],

                          

                          colLabels  = headers,

                          rowLabels = rows, 

                          rowColours = [yellow[0], yellow[1], yellow[2], green[0], green[1], green[2]] + ['lightgrey'] * 3,

                          cellLoc  = 'center',                     

                          loc = 'bottom',

                          zorder = 100,

                          bbox = [0, -0.8, 1, 0.8]

                         ) 

        

        cellDict = table.get_celld()

        for i in range(0, size):

        

            cellDict[(0,i )].set_height(.125)

        

        table.auto_set_font_size(False)

        table.set_fontsize(11)

         

    else:

        

        table = plt.table(cellText   = [female_int1, female_int2, female_int3, \

                                        male_int1,  male_int2,  male_int3],                         

                          

                          colLabels  = headers,

                          rowLabels = rows, 

                          rowColours = [yellow[0], yellow[1], yellow[2], green[0], green[1], green[2]],

                          cellLoc  = 'center',                     

                          loc = 'bottom',

                          zorder = 100, 

                          bbox = [0, -0.5, 1, 0.5]

                          

                         ) 



    

    plt.title(title, fontweight='bold')

    plt.tight_layout()
## Table 12



female_int1 = [105, 61, 49]

male_int1 = [31, 18, 24]

total_int1 = [136, 79, 73]



female_int2 = [105, 37, 43]

male_int2 = [33, 12, 15]

total_int2 = [138, 49, 58]



female_int3 = [122, 46, 40]

male_int3 = [31, 16, 12]

total_int3 = [153, 62, 52]



headers_cols12 = ['BA Hons English Literature', 'BA Hons English with CW', 'BA Hons English, \n CW and Practice']

headers_rows12 = ['Female 2015 - 2016', 'Female 2016 - 2017', 'Female 2017 - 2018',\

                  'Male 2015 - 2016', 'Male 2016 - 2017', 'Male 2017 - 2018', \

                  'Total 2015 - 2016', 'Total 2016 - 2017', 'Total 2017 - 2018']



title_12 = ""



graph_time12(female_int1, male_int1, total_int1, \

             female_int2, male_int2, total_int2, \

             female_int3, male_int3, total_int3, \

             headers_cols12, headers_rows12, title_12, 0.2, 1, True)
#Table 13



female_int1_13 = [105, 61, 28275, 49, 2725, 28275]

female_int2_13 = [105, 37, 27855, 43, 2890, 27855]

female_int3_13 = [122, 46, 26910, 40, 2990, 26910]



male_int1_13 = [31, 18, 9175, 24, 1655, 9175]

male_int2_13 = [33, 12, 8625, 15, 1720, 8625]

male_int3_13 = [31, 16, 7875, 12, 1670, 7875]



total_int1_13 = [136, 79, 37465, 73, 4385, 37465]

total_int2_13 = [138, 49, 36480, 58, 4610, 36480]

total_int3_13 = [155, 62, 34785, 52, 4660, 34785]





headers_cols13 = ['BA Hons English Literature', 'BA Hons English \n with CW', 'HESA Benchmark Q3 \n English Studies', 'BA Hons English, \n CW and Practice', \

             'HESA Benchmark W8 \n Imaginative Writing', 'HESA Benchmark Q3 \n English Studies', 'The average of \n W8 and Q3']

headers_rows13 = ['Female 2015 - 2016', 'Female 2016 - 2017', 'Female 2017 - 2018',\

                  'Male 2015 - 2016', 'Male 2016 - 2017', 'Male 2017 - 2018', \

                  'Total 2015 - 2016', 'Total 2016 - 2017', 'Total 2017 - 2018']



title_12 = ''



graph_time12(female_int1_13, male_int1_13, total_int1_13, female_int2_13, male_int2_13, total_int2_13,\

             female_int3_13, male_int3_13, total_int3_13, headers_cols13, headers_rows13, title_12, 0.25, 0, True)
#table 15



female_int1_15 = [173, 156, 44]

male_int1_15 = [44, 33, 12]

female_int2_15 = [178, 159, 41]

male_int2_15 = [47, 40, 10]

female_int3_15 = [176, 159, 43]

male_int3_15 = [50, 43, 9]

total_int1_15 = [217, 189, 56]

total_int2_15 = [225, 199, 51]

total_int3_15 = [226, 202, 52]



headers_cols15 = ['Applications', 'Offers', 'Registration']

headers_rows15 = ['Female 2015 - 2016', 'Female 2016 - 2017', 'Female 2017 - 2018',\

                  'Male 2015 - 2016', 'Male 2016 - 2017', 'Male 2017 - 2018']

title_15 = ''



graph_time12(female_int1_15, male_int1_15, total_int1_15, female_int2_15, male_int2_15, total_int2_15,\

             female_int3_15, male_int3_15, total_int3_15, \

             headers_cols15, headers_rows15, title_15, 0.2, 1, False)
#table 16



female_int1_16 = [107, 95, 19]

male_int1_16 = [36, 27, 7]

female_int2_16 = [109, 79, 14]

male_int2_16 = [26, 21, 2]

female_int3_16 = [106, 86, 12]

male_int3_16 = [26, 23, 8]

total_int1_16 = [143, 122, 26]

total_int2_16 = [135, 100, 16]

total_int3_16 = [132, 109, 20]



headers_cols16 = ['Applications', 'Offers', 'Registration']

headers_rows16 = ['Female 2015 - 2016', 'Female 2016 - 2017', 'Female 2017 - 2018',\

                  'Male 2015 - 2016', 'Male 2016 - 2017', 'Male 2017 - 2018']



title_16 = ''



graph_time12(female_int1_16, male_int1_16, total_int1_16, female_int2_16, male_int2_16, total_int2_16, \

             female_int3_16, male_int3_16, total_int3_16, \

             headers_cols16, headers_rows16, title_16, 0.2, 1, False)
#table 17



female_int1_17 = [59, 51, 19]

male_int1_17 = [14, 10, 7]

female_int2_17 = [48, 34, 15]

male_int2_17 = [12, 11, 3]

female_int3_17 = [39, 36, 19]

male_int3_17 = [12, 9, 4]

total_int1_17 = [73, 61, 26]

total_int2_17 = [60, 45, 18]

total_int3_17 = [51, 45, 23]



headers_cols17 = ['Applications', 'Offers', 'Registration']

headers_rows17 = ['Female 2015 - 2016', 'Female 2016 - 2017', 'Female 2017 - 2018',\

                  'Male 2015 - 2016', 'Male 2016 - 2017', 'Male 2017 - 2018']

title_17 = ''



graph_time12(female_int1_17, male_int1_17, total_int1_17, female_int2_17, male_int2_17, total_int2_17, \

             female_int3_17, male_int3_17, total_int3_17, \

             headers_cols17, headers_rows17, title_17, 0.2, 1, False)
def table18(firsts_fem, upper_fem, lower_fem, thirds_fem, \

            firsts_male, upper_male, lower_male, thirds_male,\

            firsts_fem_per, upper_fem_per, lower_fem_per, thirds_fem_per, \

            firsts_male_per, upper_male_per, lower_male_per, thirds_male_per, \

            headers, rows, title, bar_width, talbeNum):

     

    fem_int = [firsts_fem, upper_fem, lower_fem, thirds_fem]   

    male_int = [firsts_male, upper_male, lower_male, thirds_male]    

    

    fem_per = [firsts_fem_per, upper_fem_per, lower_fem_per, thirds_fem_per]

    male_per = [firsts_male_per, upper_male_per, lower_male_per, thirds_male_per]

   

    size = 4    

    ax = plt.subplot()    

    ax.yaxis.set_tick_params(length = 0)   

    ax.set_axisbelow(False)  

    ax.yaxis.grid(alpha = 0.2)

    ax.set_frame_on(False)    

    ax.set_ylabel('Number of students', fontsize = 14, labelpad = 15, fontweight='bold') 

    

    plt.xticks([0, 1, 2], [])

    plt.xlim(0, 3)    

    

    for i in range(3):

            

        ax.bar([3 * bar_width + x + (i * bar_width) for x in range(3)],                

                fem_per[i],

                bar_width,             

                color = yellow[i],

                edgecolor = 'black',

                linewidth = 0.5,

                label = 'Female')   

        

    for i in range(3):

            

        ax.bar([3  * bar_width + x + (i * bar_width) for x in range(3)],                

                male_per[i],

                bar_width,  

                bottom = fem_per[i],

                color = green[i],

                edgecolor = 'black',

                linewidth = 0.5,

                label = 'Female')

    i = 0    

    fem = [None] * 4    

    for pers, nums in zip(fem_per, fem_int):

        

        fem[i] = num_and_perc_format(nums, pers)

        i += 1 

        

    i = 0    

    male = [None] * 4    

    for pers, nums in zip(male_per, male_int):

        

        male[i] = num_and_perc_format(nums, pers)

        i += 1   

    

    male_bar = sum(male, [])

    fem_bar = sum(fem, [])

    

    table = plt.table(cellText   = [fem[0], male[0], \

                                    fem[1], male[1], \

                                    fem[2], male[2], \

                                    fem[3]],                       

                          

                          colLabels  = headers,

                          rowLabels = rows, 

                          rowColours = [yellow[0], green[0], yellow[1], green[1], yellow[2], green[2], 'white'],

                          cellLoc  = 'center',                     

                          loc = 'bottom',

                          zorder = 100,

                          bbox = [0, -0.7, 1, 0.7]

                          ) 

   

    plt.title(title, fontweight = 'bold')

    plt.tight_layout()
#table 18 - remove last row



firsts_fem = [18, 16, 20]

first_fem_per = [72, 70, 74]

upper_fem = [50, 31, 47]

upper_fem_per = [70, 76, 75]

lower_fem = [5, 1, 1]

lower_fem_per = [100, 25, 50]

thirds_fem = [0, 1, 0]

thirds_fem_per = [0, 100, 0]



firsts_male = [7, 7, 7]

firsts_male_per = [28, 30, 26]

upper_male = [21, 10, 16]

upper_male_per = [30, 24, 25]

lower_male = [0, 3, 1]

lower_male_per = [0, 75, 50]

thirds_male = [0, 0, 0]

thirds_male_per = [0, 0, 0]



headers_cols_18 = ['2015 - 16', '2016 - 17', '2017 - 18']

headers_rows_18 = ['1st Female', '1st Male', '2:1 Female', '2:1 Male','2:2 Female', '2:2 Male', '3rd/Pass Female']                



title_18 = ''



table18(firsts_fem, upper_fem, lower_fem, thirds_fem, \

        firsts_male, upper_male, lower_male, thirds_male, \

        first_fem_per, upper_fem_per, lower_fem_per, thirds_fem_per, \

        firsts_male_per, upper_male_per, lower_male_per, thirds_male_per, \

        headers_cols_18, headers_rows_18, title_18, 0.125, False)
female_int1_20 = [23, 31, 54]

female_int2_20 = [28, 23, 51]

female_int3_20 = [24, 25, 49]

male_int1_20 = [9, 5, 14]

male_int2_20 = [10, 8, 18]

male_int3_20 = [18, 13, 31]

total_int1_20 = [32, 36, 68]

total_int2_20 = [38, 31, 69]

total_int3_20 = [42, 38, 80]



headers_cols20 = ['FT', 'PT', 'Total', 'Ratio FT:PT']

headers_rows20 = ['Female 2015 - 2016', 'Female 2016 - 2017', 'Female 2017 - 2018',\

                  'Male 2015 - 2016', 'Male 2016 - 2017', 'Male 2017 - 2018']



title_20 = ''



graph_time12(female_int1_20, male_int1_20, total_int1_20, female_int2_20, male_int2_20, total_int2_20, \

             female_int3_20, male_int3_20, total_int3_20, \

             headers_cols20, headers_rows20, title_20, 0.2, 1, False)
def table_21(female_ft_20, male_ft_20, ft_total_20, \

             female_pt_20, male_pt_20, pt_total_20,\

             headers_rows_20, headers_cols_main_20, headers_cols_20, bar_width, title):

    

    offset = 1.5

    

    size = len(female_ft_20)

    plt.figure(figsize = (12, 5))

    if(size == 3): plt.figure(figsize = (8, 7))

        

    elif(size == 12): 

        

        plt.figure(figsize = (15.5, 7))

        offset = 0.625

    

    elif(size == 4):

        

        plt.figure(figsize = (6, 5))

        offset = 0.6

    

    ft_colors = [yellow[0], green[0]]

    pt_colors = [yellow[1], green[1]]

    

    full_time = [female_ft_20, male_ft_20]

    full_time_per = [calc_perc_float(female_ft_20, ft_total_20), calc_perc_float(male_ft_20, ft_total_20)]

    

    part_time = [female_pt_20, male_pt_20]

    part_time_per = [calc_perc_float(female_pt_20, pt_total_20), calc_perc_float(male_pt_20, pt_total_20)]

    

    total = female_ft_20 + male_ft_20 + female_pt_20 + male_pt_20

    total_per = [j for sub in full_time_per + part_time_per for j in sub]

    

    per_plus_nums = num_and_perc_format(total, total_per)

    

    ax = plt.subplot()  

    ax.set_ylim(0, 110)

    ax.yaxis.set_tick_params(length = 0)   

    ax.set_axisbelow(False)  

    ax.grid(True, which="both",ls="dashed", color='0.01', alpha = 0.1)

    ax.set_frame_on(False)    

    ax.set_ylabel('% of students', fontsize = 14, labelpad = 15, fontweight='bold') 

    ax.set_xlim(0, size)

    ax.set_xticks([], [])

    

    for i, array in enumerate(full_time_per):

        

        btm = 0

        

        if(i > 0): btm = full_time_per[0]

        

        ax.bar([offset * bar_width + x for x in range(size)],                       

               array,

               bar_width,     

               bottom = btm,

               color = ft_colors[i],

               edgecolor = 'black',

               linewidth = 0.5,

               label = 'Female') 

    

    if(size != 12 and size != 4):

        

        for i, array in enumerate(part_time_per):

        

            btm = 0



            if(i > 0): btm = part_time_per[0]



            ax.bar([bar_width + x + 2.5 * bar_width for x in range(size)],                       

                   array,

                   bar_width,     

                   bottom = btm,

                   color = pt_colors[i],

                   edgecolor = 'black',

                   linewidth = 0.5,

                   label = 'Male') 

    

    bbox_next = [0, -0.35, 1, 0.35]

    

    if size != 3:

        

        plt.table(cellText   = [headers_cols_main_20],                       

                  cellLoc  = 'center',                     

                  loc = 'bottom',

                  zorder = 100,

                  edges = 'closed',

                  alpha = 0.5,

                  bbox = [0, -0.2, 1, -0.1]

                ) 

        

        bbox_next = [0, -0.68, 1, -.082]

        

    if (size == 12 or size == 4) and title != None:

        

        if(size == 4): 

            

            bbox_next = [0, -0.5, 1, 0.4]

      

        else: bbox_next = [0, -0.3, 1, -.06]

        

        table = plt.table(cellText = [per_plus_nums[0:size], per_plus_nums[size:size*2], ft_total_20],

                      colLabels = headers_cols_20,

                      rowLabels = headers_rows_20,

                      rowColours = [ft_colors[0], ft_colors[1], 'lightgrey'],

                      cellLoc = 'center',

                      loc ='bottom',

                      bbox = bbox_next,

                    )

        

        table.auto_set_font_size(False)

        table.set_fontsize(10)

        

    elif title == None:

        

        table = plt.table(cellText = [per_plus_nums[0:size], per_plus_nums[size:size*2], ft_total_20],

                      colLabels = headers_cols_20,

                      rowLabels = headers_rows_20,

                      rowColours = [ft_colors[0], ft_colors[1], 'lightgrey'],

                      cellLoc = 'center',

                      loc ='bottom',

                      bbox= [0, -0.35, 1, -.06],

                    )

        

        table.auto_set_font_size(False)

        table.set_fontsize(10)

    

        

    else:

    

        print('ok')

        plt.table(cellText = [per_plus_nums[0:size], per_plus_nums[size:size*2], ft_total_20, \

                              per_plus_nums[size*2:size*3], per_plus_nums[size*3:size*4], pt_total_20],

                      colLabels = headers_cols_20,

                      rowLabels = headers_rows_20,

                      rowColours = [ft_colors[0], ft_colors[1], 'lightgrey', pt_colors[0], pt_colors[1], 'lightgrey'],

                      cellLoc = 'center',

                      loc ='bottom',

                      bbox= bbox_next,

                    )

   

    plt.title(title, fontweight='bold')

    plt.tight_layout()

    plt.show()
#Table 21



female_ft_20 = [4, 560, 6, 735, 16, 770]

male_ft_20 = [5, 250, 5, 410, 11, 410]

ft_total_20 = [x + y for x, y in zip(female_ft_20, male_ft_20)]



female_pt_20 = [26, 670, 22, 975, 22, 120]

male_pt_20 = [5, 330, 7, 460, 11, 490]

pt_total = [x + y for x, y in zip(male_pt_20, female_pt_20)]



headers_rows_20 = ['Female FT', 'Male FT', 'Total FT'] * 2

headers_cols_main_20 = ['2015 - 16', '2016 - 17', '2017 - 18']

headers_cols_20 = ['CW PGT', 'HESA W8'] * 3

bar_width = 0.2

title_20 = ''



table_21(female_ft_20, male_ft_20, ft_total_20, \

         female_pt_20, male_pt_20, pt_total, \

         headers_rows_20, headers_cols_main_20, headers_cols_20, bar_width, title_20)

#Table 22



female_ft_20 = [19, 1315, 22, 1545, 8, 1820]

male_ft_20 = [4, 500, 5, 655, 7, 710]

ft_total_20 = [x + y for x, y in zip(female_ft_20, male_ft_20)]



female_pt_20 = [5, 985, 2, 1060, 3, 1030]

male_pt_20 = [0, 350, 1, 355, 2, 405]

pt_total = [x + y for x, y in zip(male_pt_20, female_pt_20)]



headers_rows_20 = ['Female FT', 'Male FT', 'Total FT', 'Female PT', 'Male PT', 'Total PT']

headers_cols_main_20 = ['2015 - 16', '2016 - 17', '2017 - 18']

headers_cols_20 = ['CW PGT', 'HESA Q3'] * 3

bar_width = 0.2

title_20 = ''



table_21(female_ft_20, male_ft_20, ft_total_20, \

         female_pt_20, male_pt_20, pt_total, \

         headers_rows_20, headers_cols_main_20, headers_cols_20, bar_width, title_20)
#table 23 - remove last row



app_fem = [25, 34, 45]

app_fem_per = [61, 65, 60]

off_fem = [10, 15, 32]

off_fem_per = [45, 63, 60]

reg_fem = [4, 7, 17]

reg_fem_per = [50, 58, 61]



app_male = [10, 11, 18]

app_male_per = [39, 35, 40]

off_male = [7, 9, 11]

off_male_per = [55, 37, 40]

reg_male = [2, 7, 6]

reg_male_per = [50, 42, 39]





headers_cols_18 = ['2015 - 16', '2016 - 17', '2017 - 18']

headers_rows_18 = ['Applications Female', 'Applications Male', 'Offers Female', 'Offers Male','Registrations Female', 'Registrations Male', '3rd/Pass Female']                



title_18 = ''



table18(app_fem, off_fem, reg_fem, app_fem, \

        app_male, off_male, reg_male, app_fem, \

        app_fem_per, off_fem_per, reg_fem_per, app_fem, \

        app_male_per, off_male_per, reg_male_per, app_fem, \

        headers_cols_18, headers_rows_18, title_18, 0.125, 23)
#table 24



app_fem = [38, 19, 30]

app_fem_per = [79, 63, 63]

off_fem = [23, 17, 23]

off_fem_per = [79, 65, 68]

reg_fem = [13, 8, 13]

reg_fem_per = [87, 53, 68]



app_male = [10, 11, 18]

app_male_per = [21, 37, 37]

off_male = [7, 9, 11]

off_male_per = [21, 35, 32]

reg_male = [2, 7, 6]

reg_male_per = [13, 47, 32]





headers_cols_18 = ['2015 - 16', '2016 - 17', '2017 - 18']

headers_rows_18 = ['Applications Female', 'Applications Male', 'Offers Female', 'Offers Male','Registrations Female', 'Registrations Male', '3rd/Pass Female']                



title_18 = ''



table18(app_fem, off_fem, reg_fem, app_fem, \

        app_male, off_male, reg_male, app_fem, \

        app_fem_per, off_fem_per, reg_fem_per, app_fem, \

        app_male_per, off_male_per, reg_male_per, app_fem, \

        headers_cols_18, headers_rows_18, title_18, 0.125, 24)
#table 25



app_fem = [69, 57, 50]

app_fem_per = [77, 71, 69]

off_fem = [39, 43, 37]

off_fem_per = [75, 74, 74]

reg_fem = [13, 8, 13]

reg_fem_per = [87, 53, 68]



app_male = [21, 23, 22]

app_male_per = [23, 29, 31]

off_male = [13, 15, 13]

off_male_per = [25, 26, 26]

reg_male = [2, 7, 6]

reg_male_per = [13, 47, 32]





headers_cols_18 = ['2015 - 16', '2016 - 17', '2017 - 18']

headers_rows_18 = ['Applications Female', 'Applications Male', 'Offers Female', 'Offers Male','Registrations Female', 'Registrations Male', '3rd/Pass Female']                



title_18 = ''



table18(app_fem, off_fem, reg_fem, app_fem, \

        app_male, off_male, reg_male, app_fem, \

        app_fem_per, off_fem_per, reg_fem_per, app_fem, \

        app_male_per, off_male_per, reg_male_per, app_fem, \

        headers_cols_18, headers_rows_18, title_18, 0.125, 24)
#Table 26



female_ft_26 = [11, 9, 4, 7, 12, 1, 9, 15, 2]

male_ft_26 = [2, 3, 2, 2, 3, 1, 5, 4, 0]

ft_total_26 = [x + y for x, y in zip(female_ft_26, male_ft_26)]



female_pt_26 = [5, 8, 4, 7, 9, 0, 5, 6, 1]

male_pt_26 = [1, 5, 0, 1, 2, 0, 0, 1, 1]

pt_total_26 = [x + y for x, y in zip(male_pt_26, female_pt_26)]



headers_rows_26 = ['Female FT', 'Male FT', 'Total FT', 'Female PT', 'Male PT', 'Total PT']

headers_cols_main_26 = ['2015 - 16', '2016 - 17', '2017 - 18']

headers_cols_26 = ['Pass with \n distinction', 'Pass with Merit', 'Pass'] * 3

bar_width = 0.2

title_26 = ''



table_21(female_ft_26, male_ft_26, ft_total_26, \

         female_pt_26, male_pt_26, pt_total_26, \

         headers_rows_26, headers_cols_main_26, headers_cols_26, bar_width, title_26)

    
#Table 28



female_int1_28 = [21, 12]

female_int2_28 = [24, 6]

female_int3_28 = [13, 3]

male_int1_28 = [10, 4]

male_int2_28 = [10, 3]

male_int3_28 = [6, 3]

total_int1_28 = [31, 16]

total_int2_28 = [34, 9]

total_int3_28 = [19, 6]



headers_cols28 = ['FT', 'PT']

headers_rows28 = ['Female 2015 - 2016', 'Female 2016 - 2017', 'Female 2017 - 2018',\

                  'Male 2015 - 2016', 'Male 2016 - 2017', 'Male 2017 - 2018']



title_28 = ''



graph_time12(female_int1_28, male_int1_28, total_int1_28, female_int2_28, male_int2_28, total_int2_28, \

             female_int3_28, male_int3_28, total_int3_28, \

             headers_cols28, headers_rows28, title_28, 0.2, 3, False)
female_int1_29 = [8, 7]

female_int2_29 = [8, 9]

female_int3_29 = [12, 8]



male_int1_29 = [3, 1]

male_int2_29 = [6, 1]

male_int3_29 = [2, 2]



total_int1_29 = [11, 8]

total_int2_29 = [14, 10]

total_int3_29 = [14, 10]



headers_cols29 = ['FT', 'PT']

headers_rows29 = ['Female 2015 - 2016', 'Female 2016 - 2017', 'Female 2017 - 2018',\

                  'Male 2015 - 2016', 'Male 2016 - 2017', 'Male 2017 - 2018']



title_29 = 'Table 29'



graph_time12(female_int1_29, male_int1_29, total_int1_29, female_int2_29, male_int2_29, total_int2_29, female_int3_29, male_int3_29, total_int3_29, \

             headers_cols29, headers_rows29, title_29, 0.2, 2, False)
#Table 30



female_ft_30 = [8, 115, 8, 120, 12, 130]

male_ft_30 = [3, 40, 6, 50, 2, 65]

ft_total_30 = [x + y for x, y in zip(female_ft_30, male_ft_30)]



female_pt_30 = [26, 670, 22, 975, 22, 120]

male_pt_30 = [1, 55, 1, 50, 3, 50]

pt_total = [x + y for x, y in zip(male_pt_30, female_pt_30)]



headers_rows_30 = ['Female FT', 'Male FT', 'Total FT', 'Female PT', 'Male PT', 'Total PT']

headers_cols_main_30 = ['2015 - 16', '2016 - 17', '2017 - 18']

headers_cols_30 = ['ELCW PGR CW', 'HESA W8'] * 3

bar_width = 0.2

title_30 = 'Table 30'



table_21(female_ft_30, male_ft_30, ft_total_30, \

         female_pt_30, male_pt_30, pt_total, \

         headers_rows_30, headers_cols_main_30, headers_cols_30, bar_width, title_30)
#Table 32



female_ft_30 = [63, 36, 12, 47, 24, 7, 52, 18, 8]

male_ft_30 = [34, 18, 4, 21, 12, 2, 29, 8, 2]

ft_total_30 = [x + y for x, y in zip(female_ft_30, male_ft_30)]



female_pt_30 = [9, 6, 7, 12, 5, 2, 5, 2, 2]

male_pt_30 = [5, 0, 1, 2, 2, 1, 5, 2, 2]

pt_total = [x + y for x, y in zip(male_pt_30, female_pt_30)]



headers_rows_30 = ['Female FT', 'Male FT', 'Total FT', 'Female PT', 'Male PT', 'Total PT']

headers_cols_main_30 = ['2015 - 16', '2016 - 17', '2017 - 18']

headers_cols_30 = ['Applications', 'Offers', 'Registrations'] * 3

bar_width = 0.2

title_30 = 'Table 32'



table_21(female_ft_30, male_ft_30, ft_total_30, \

         female_pt_30, male_pt_30, pt_total, \

         headers_rows_30, headers_cols_main_30, headers_cols_30, bar_width, title_30)

    
#Table 33



female_ft_30 = [4, 4, 3]

male_ft_30 = [2, 3, 4]

ft_total_30 = [x + y for x, y in zip(female_ft_30, male_ft_30)]



female_pt_30 = [4, 1, 0]

male_pt_30 = [1, 0, 1]

pt_total = [x + y for x, y in zip(male_pt_30, female_pt_30)]



headers_rows_30 = ['Female FT', 'Male FT', 'Total FT', 'Female PT', 'Male PT', 'Total PT']

headers_cols_main_30 = ['2011 - 12', '2012 - 13', '2013 - 14']

headers_cols_30 = ['2011 - 12', '2012 - 13', '2013 - 14']

bar_width = 0.2

title_30 = 'Table 33'



table_21(female_ft_30, male_ft_30, ft_total_30, \

         female_pt_30, male_pt_30, pt_total, \

         headers_rows_30, headers_cols_main_30, headers_cols_30, bar_width, title_30)

    
#Table 34



female_ft_30 = [9, 3, 3, 3, 5, 3, 3, 3, 7, 2, 2, 1]

male_ft_30 = [5, 2, 1, 1, 2, 1, 1, 0, 5, 3, 3, 3]

ft_total_30 = [x + y for x, y in zip(female_ft_30, male_ft_30)]



headers_rows_30 = ['Female', 'Male', 'Total']

headers_cols_main_30 = ['2015 - 16', '2016 - 17', '2017 - 18']

headers_cols_30 = ['Applications', 'Shorlisted', 'Offered funding', 'Took up funding'] * 3

bar_width = 0.2

title_30 = None



table_21(female_ft_30, male_ft_30, ft_total_30, \

         [], [], [], \

         headers_rows_30, headers_cols_main_30, headers_cols_30, 0.75, title_30)
#Table 34



female_ft_30 = [1, 15, 16, 2]

male_ft_30 = [7, 10, 16, 2]

ft_total_30 = [x + y for x, y in zip(female_ft_30, male_ft_30)]



headers_rows_30 = ['Female', 'Male', 'Total']

headers_cols_main_30 = ['REF 2014', 'RAE 2008']

headers_cols_30 = ['Included', 'Not included', 'Category A', 'Category B']

bar_width = 0.2

title_30 = 'Table 49'



table_21(female_ft_30, male_ft_30, ft_total_30, \

         [], [], [], \

         headers_rows_30, headers_cols_main_30, headers_cols_30, 0.75, title_30)
##Table 56



fem_total2 = [11, 6, 9]

male_total2 = [10, 15, 12]

fem_per2 = [52, 29, 43]

male_per2 = [48, 71, 57]

total2 = [21, 21, 21]



headers_cols56 = ['2015 - 16', '2016 - 17', '2017 - 18']

title2 = 'Table 56'



graph_bar_table2(fem_total2, male_total2, fem_per2, male_per2, total2, headers_cols56, headers_rows2, title2)
def table_59(int1, int2, int3, total, headers_cols, headers_rows, title, bar_width):

    

    full = [int1, int2, int3]

    size = 4

    plt.figure(figsize = (8, 4))

    

    ax = plt.subplot()    

    ax.yaxis.set_tick_params(length = 0)   

    ax.set_axisbelow(False)  

    ax.yaxis.grid(alpha = 0.2)

    ax.set_frame_on(False)    

    ax.set_ylabel('Number of staff', fontsize = 14, labelpad = 15, fontweight='bold') 

    

    plt.xticks([], [])

    plt.xlim(0, size)

    

    for i in range(3):

            

        ax.bar([4 * bar_width + x + (i * bar_width) for x in range(size)],                

                full[i],

                bar_width,             

                color = others[i],

                edgecolor = 'black',

                linewidth = 0.5,

                label = 'Female')  

        

    print([int1, int2, int3, total])

        

    plt.table(cellText = [int1, int2, int3, total],

                      colLabels = headers_cols,

                      rowLabels = headers_rows,

                      cellLoc = 'center',

                      loc ='bottom',

                      bbox = [0, -0.4, 1, 0.4]

               )

    

    

    plt.title(title, fontweight='bold')

    plt.show()

    plt.savefig('X.png')
## Table 59



int1 = [10, 7, 2, 1]

int2 = [8, 6, 6, 1]

int3 = [6, 10, 0, 5]

total = ['24(38%)', '23(37%)', '8(12%)', '7(11%)']



headers_cols59 = ['Lecturer', 'Senior lecturer', 'Reader', 'Professor']

headers_rows59 = ['2015 - 16', '2016 - 17', '2017 - 18', 'Total']

bar_width = 0.1



title_59 = "Table 57"



table_59(int1, int2, int3, total, headers_cols59, headers_rows59, title_59, bar_width)
## Table 59



int1 = [6, 4, 2, 3]

int2 = [5, 4, 3, 3]

int3 = [8, 5, 0, 4]

total = ['19(40%)', '13(27%)', '5(11%)', '10(22%)']



headers_cols59 = ['Lecturer', 'Senior lecturer', 'Reader', 'Professor']

headers_rows59 = ['2015 - 16', '2016 - 17', '2017 - 18', 'Total']

bar_width = 0.1



title_59 = ""



table_59(int1, int2, int3, total, headers_cols59, headers_rows59, title_59, bar_width)
## Table 61



int1 = [1, 9, 2, 5]

int2 = [6, 5, 3, 4]

int3 = [9, 6, 0, 3]

total = ['16(30%)', '20(38%)', '5(10%)', '12(22%)']



headers_cols59 = ['Lecturer', 'Senior lecturer', 'Reader', 'Professor']

headers_rows59 = ['2015 - 16', '2016 - 17', '2017 - 18', 'Total']

bar_width = 0.1



title_59 = ""



table_59(int1, int2, int3, total, headers_cols59, headers_rows59, title_59, bar_width)
## Table 63



int1 = [0, 2, 0, 6]

int2 = [0, 2, 0, 6]

int3 = [0, 0, 0, 8]

total = ['0', '14(17%)', '0', '20(83%)']



headers_cols59 = ['Lecturer', 'Senior lecturer', 'Reader', 'Professor']

headers_rows59 = ['2015 - 16', '2016 - 17', '2017 - 18', 'Total']

bar_width = 0.1



title_59 = ""



table_59(int1, int2, int3, total, headers_cols59, headers_rows59, title_59, bar_width)