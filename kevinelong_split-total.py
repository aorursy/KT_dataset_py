text = "123.11-456.22-789.99"
# convert from text to list of string
list_of_strings = text.split("-")

total = 0

for t in list_of_strings:
    value = float(t) # CONVERT text fragment into integer
    total += value
    
print(total)

