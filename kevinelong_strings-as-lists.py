name = "KEVIN"
for letter in name:
    print(letter)
# Ordinal Value (e.g. Number) ASCII American Standard Code for Infromation Interchange
print(ord("A"))
print(ord("a"))

print(ord("B"))
print(ord("b"))


# GOAL SAMPLE OUTPUT
# A 65
# B 66
# C 67
name= "KEVIN"
for letter in name:
    number = ord(letter)
    print(letter, end=" ")
    print(number, end=" ")
    
    print(bin(number), end=" ")
    print(hex(number), end=" ")
    
    print("")
    
x = 0b1000001
print(x)
print(chr(x))
data = [
    0b1001011,  
    0b1000101, 
    0b1010110, 
    0b1001001,  
    0b1001110
]
for n in data:
    print(chr(n))