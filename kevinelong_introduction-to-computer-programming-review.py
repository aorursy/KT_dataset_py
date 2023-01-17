import multiprocessing
multiprocessing.cpu_count()
from psutil import virtual_memory
print(f"{virtual_memory().total:,}")
import shutil

total, used, free = shutil.disk_usage("/")

print(f"Total: {total:,}")
print(f"Used: {used:,}")
print(f"Free: {free:,}")
import sys
x = 1
sys.getsizeof(x)

inventory_quantity = 123
text = ""
sys.getsizeof(text)
letter = "A"
ord("A")
ord("B")
ord(" ") 
name = "A a B b"
for letter in name:
    print(letter, ord(letter), bin(ord(letter)))


for n in range(32, 256):
    print(chr(n), n, bin(n))
print("😀", ord("😀"), bin(ord("😀")))

# a😀 = "Kevin"

# print(a😀)
number = ord("😀")
#number = 120512 

for offset in range(64):
    print(number + offset, chr(number + offset))
    
