import sys
print(sys.argv)

for a in sys.argv:
    print(a)

number = sys.argv[1]
file = sys.argv[2]
print(f"{number} times will be output to: {file}")

