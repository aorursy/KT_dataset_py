def read_codons():
    f = open('../input/codons/codons.txt', 'r') # Imports the relevant file in read mode using 'r'
    raw_data = f.read()                         # Reads the file as a string, and saves the whole thing as a string
    lines = raw_data.split('\n')                # Makes a list of lines divided by \n

    # this is the same as the fruits code from class
    codons = {}
    for line in lines:
        columns = line.split()
        codons[columns[0]] = columns[1]
    return codons

def mrna_to_protein(mrna_in):
    protein_out = ""                # make a string to add on to
    num_codons = len(mrna_in)//3    # there are 3 bases per codon, so I'm taking the length of my string divided by 3 and rounding down to get the number to look for
    for i in range(num_codons): 
        codon_start = 3*i                            # Each codon is spaced 3 characters apart
        codon_end   = codon_start + 3                # Each codon is 3 bases long, so it ends at start + e
        this_codon  = mrna_in[codon_start:codon_end] # Get the 3-character substring from the input
        new_AA      = codons[this_codon]             # Look up the codon in our dictionary
        if new_AA == 'Stop':                         # Don't append the word "Stop" to the end
            break
        protein_out += new_AA                        # Add the new amino acid to our protein string
    return protein_out

def dna_to_mrna(dna_in):
    return dna_in.replace("T","U") # From the previous problems

def dna_to_protein(dna_in):
    mrna = dna_to_mrna(dna_in)     # Convert DNA into RNA
    return mrna_to_protein(mrna)   # Run mrna_to_protein on the converted string and return the result


codons = read_codons()
dna_to_protein("AATCTCTACGGAAGTAGGTCAGTACTGATCGATCAGTCGATCGGGCGGCGATTTCGATCTGATTGTACGGCGGGCTAG")
def read_fasta(file_in):
    f = open(file_in, 'r')      # Imports the relevant file in read mode using 'r'
    raw_data = f.read()         # Reads the file as a string, and saves the whole thing as a string
    lines = raw_data.split('>') # Makes a list of lines divided by >
    
    d = {}                      # An empty dictionary to feed results into
    for line in lines[1:]:      # Loop through every entry. Because it starts with >, we can skip the first empty entry in lines
        name,seq = line.split('\n',1) # Use split to pull off the first entry before a \n as name, and the rest as seq
        seq = seq.replace('\n','')    # Get rid of all the rest of the \n characters
        d[name]=seq             # save the name-->seq pair
    return d

dnas = read_fasta('../input/codons/PS2Q2.fasta')

for dna_key in dnas:
    print(dna_key+": "+dna_to_protein(dnas[dna_key]))

