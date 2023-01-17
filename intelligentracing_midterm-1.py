#Midium1 ex3_String.py

#Author: Yu Qiuhsuang



print('please input the orign sentence:')

orign = input()

print('please input the test sentence:')

test = input()





orign = orign.replace(' ','')

orign = sorted(orign)

orign = [s.lower() for s in orign if isinstance(s,str)==True]

test = test.replace(' ','')

test = sorted(test)

test = [s.lower() for s in test if isinstance(s,str)==True]

# test_result = []

# for s in test:

#     if isinstance(s, str) == True:

#         test_result.append(s.lower()) 

all = [x for x in orign if x in test]

result = [y for y in (orign + test) if y not in all]



if len(result) == 0:

    is_anagram = True

else:

    is_anagram = False



print(is_anagram)
#Midium1 ex4_List.py

#Author: Yu Qiuhsuang



print('please input the number for the list without any marks or empty spaces:')

element = input()

print('please input shift steps:')

shift = input()

shift = int(shift)

test_list = list(element)

result_list = []

for i in test_list:

    result_list.append(int(i))



n = len(result_list)

result_list_copy = result_list.copy()

for s in range(n):

    if s + shift < 0:

        result_list_copy[n + s + shift] = result_list[s]

    if n > s + shift >= 0:

        result_list_copy[s + shift] = result_list[s]

    if s + shift >= n:

        result_list_copy[s + shift - n] = result_list[s]

        

print(result_list_copy)
#Midium1 ex5_print_text_patterns.py

#Author: Yu Qiuhsuang



print('please input times range from 0 to 10:')

times = input()

times = int(times)

for i in range(-10,11):

    print('*', end ='')

print()

#整个沙漏从上向下一行一行打印，从-10，打印到10，从上向下每一行依次是-10， -9， -8，......

for i in range(-10, 11):

    a = ' '	

    if i < -10 + times or i > 10-(10 -times):

        a ='.' 

    # 最前边的空格

    for j in range(10 - abs(i)):

        print(' ', end='')

    # *

    for k in range(2 * abs(i) + 1):

        #首尾的*

        if k == 0 or k == 2 * abs(i):

            print('*', end='')

        else:

            #中间的*

            print(a, end ='')

    print()



for i in range(-10,11):

    print('*', end ='')

print()