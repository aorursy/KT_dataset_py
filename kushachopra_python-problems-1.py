input_dna = "placeholder"
input_dna = input("your DNA sequence")
for ch in input_dna:
    input_mRNA = input_dna.replace ("T", "U")
print(input_mRNA)

    


input_dna = "placeholder"
input_dna = input("your DNA sequence")
cDNA = ""
for ch in input_dna[::-1]:
    if ch == "A":
        cDNA = cDNA + "T"
    if ch == "T":
        cDNA = cDNA + "A"
    if ch == "G":
        cDNA = cDNA + "C"
    if ch == "C":
        cDNA = cDNA + "G"
        
print(cDNA)

input_dna = "placeholder"
input_dna = "GATTCTCTGGAGAGAAGCTTCTCTCCAGAGAATC"
cDNA = ""
for ch in input_dna[::-1]:
    if ch == "A":
        cDNA = cDNA + "T"
    if ch == "T":
        cDNA = cDNA + "A"
    if ch == "G":
        cDNA = cDNA + "C"
    if ch == "C":
        cDNA = cDNA + "G"
        
if cDNA == input_dna:
    print("true")
elif cDNA != input_dna:
    print("false")