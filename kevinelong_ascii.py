# ASCII

# 65 == "A"
# 97 == "a"

print( ord("b") ) #ordinal integer ascii

print( chr(64) ) # character for an integer
# 3 == ^C
for i in range(32,128):
    print(i, bin(97), chr(i), end="\t")
    
print(bin(97))
def numbers(v):
    print(v, bin(v), hex(v), chr(v))


def show_ascii(name):
    for letter in name:
        ascii_code = ord(letter)
        numbers(ascii_code)


show_ascii("KEVIN")

r = 200
g = 240
b = 64

rgb = [r, g, b]

for v in rgb:
    numbers(v)

