import string
import random
l_alphabets = list(string.ascii_lowercase)
u_alphabets = list(string.ascii_uppercase)
digits = list(string.digits)
punct = ['!','@','$','#','^','&','*']
lst = []
lst.extend(u_alphabets)
lst.extend(l_alphabets)
lst.extend(digits)
lst.extend(punct)
pwd = ''
for i in range(12):
    idx = random.randint(0,len(lst)-1)
    pwd += (lst[idx])
print(pwd)