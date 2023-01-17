lst = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1] 
otherlst = ['a','b','c','d','e','f','g']
s = "This is a test string for HCDE 530"

#Exercise 1 (working with a list):
#a.	Print the first element of lst (this one has been completed for you)
print(lst[0])

#b.	Print the last element of otherlst
print(otherlst[0])

#c.	Print the first five elements of lst
print(lst[0])
print(lst[1])
print(lst[2])
print(lst[3])
print(lst[4])

#d.	Print the fifth element of otherlst
print(otherlst[4])

#e.	Print the number of items in lst
print(len(lst))

#Exercise 2 (working with a string):
#a.	Print the first four characters of s
print(s[:4])

#b.	Using indexing, print the substring "test" from s
print(s[10:14])

#c.	Print the contents of s starting from the 27th character (H)
print(s[26:])
#this question is unclear. Do you want to include H or start from the 28th character? I assumed you wanted to include 'H'.

#d.	Print the last three characters of s
print(s[-3:])

#e.	Print the number of characters in s
print(len(s))
n = 13
fact = 1
  
for i in range(1,n+1): 
    fact = fact * i 
      
print (fact) 
word_1 = "Happy " 
word_2 = "New " 
word_3 = "Year!"

print(word_1 + word_2 + word_3)
word_1 = "Happy " 
word_2 = "New " 
word_3 = "Year!"

print(word_1)
print(word_2)
print(word_3)


Part_5 = [word_1, word_2, word_3]
CombinedString = " "
print(CombinedString.join(Part_5))
value_1 = (3, 4)
sum_value= sum(value_1)
print(sum_value)