set_num1 = {1,2,3,4,5,6}
set_num2 = {5,6,8,1,2}
set_num1 - set_num2 #will give numbers that are only in set_num1
set_num1.difference(set_num2) #will give numbers that are only in set_num1
set_num2 - set_num1 #will give numbers that are only in set_num2
set_num2.difference(set_num1) #will give numbers that are only in set_num2
set_num1 | set_num2 #will give only not repeted numbers from set_num1 and y
set_num1.union(set_num2)
set_num2 & set_num1 #will give only repeted numbers from set_num1 and set_num2
set_num1.intersection(set_num2)
set_num1.add(10) # add number to the set
set_num1
set_str1 = {'a','s','d','f'}
set_str2 = {'g','s','d','f'}
set_str1 - set_str2 #will give string that are only in set_str1
set_str1.difference(set_str2) #will give string that are only in set_str1
set_str2 - set_str1 #will give string that are only in set_str1
set_str2.difference(set_str1) #will give string that are only in set_str1
set_str1 | set_str2 #will give only not repeted string from set_str1 and set_str2
set_str1.union(set_str2) #will give only not repeted string from set_str1 and set_str2
set_str1.intersection(set_str2)
set_str1.add('h')
set_str1
set_str1.issubset(set_str2) # to check if the both set are the same values
set_str1 == set_str2