print("X", end="")
print("X", end="")
print("X", end="")
print("")

print("X", end="")
print("X", end="")
print("X", end="")
print("")

for i in range(5):
    print(i)
    
# L1 
# XXXXXX

# L2 
# X
# XX
# XXX
# XXXX
# XXXXX
# XXXXXX

#L3
#      X
#     XX
#    XXX
#   XXXX
#  XXXXX
# XXXXXX


# L1 
# XXXXXX
for index in range(6):
    print("X", end="")
print("")
# L2 
# X
# XX
# XXX
# XXXX
# XXXXX
# XXXXXX
limit = 6
for i in range(1,limit+1):
    for x in range(i):
        print("X", end="")
    print()

#L3
#      X
#     XX
#    XXX
#   XXXX
#  XXXXX
# XXXXXX
limit = 6
for i in range(0,limit):
    for x in range(limit - i - 1):
        print(" ", end="")
    for x in range(i + 1):
        print("X", end="")
    print()
# Path seperators
# \ dos/window
# / everyone else