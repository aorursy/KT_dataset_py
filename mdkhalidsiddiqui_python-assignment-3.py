sub_code1 = input("Enter the first subject code:")
sub_title1 = input("Enter the first subject title:")
sub_code2 = input("Enter the second subject code:")
sub_title2 = input("Enter the second subject title:")
sub_code3 = input("Enter the third subject code:")
sub_title3 = input("Enter the third subject title:")
print("{0:<15}{1:<40}".format("Code","Title"))
print("{0:<15}{1:<40}".format(sub_code1,sub_title1))
print("{0:<15}{1:<40}".format(sub_code2,sub_title2))
print("{0:<15}{1:<40}".format(sub_code3,sub_title3))
ip1 = input("Enter the first number: ")
num1=int(ip1)
ip2 = input("Enter the second number: ")
num2=int(ip2)
print("The sequence is:  "+ str(num1) \
      +", "\
      + str(num2) \
      + ", "\
      + str(num1 + num2) \
      + ", " \
      + str(num2 + num1 + num2) \
      +", " \
      + str(num2 + num1 + num2 + num1 + num2) \
      + ", "\
      + "...")
ip1 = input("Enter the first number: ")
num1=int(ip1)
ip2 = input("Enter the second number: ")
num2=int(ip2)

print ("The sequence is: {0}, {1}, {2}, {3}, {4}, ...".format(num1,num2,num1+num2,num2+num1+num2,num1+num2+num2+num1+num2))