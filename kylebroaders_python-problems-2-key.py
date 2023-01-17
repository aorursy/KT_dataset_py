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


codons=read_codons()
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
# The following code builds 4 lists which track the numbers of each base. It then takes the highest value among the 4 lists to be the 'consensus' base for that position
# Make print statements in this cell or in a new cell below to explore what's in each variable.

dnas = read_fasta('../input/codons/PS2Q3.fasta')
seq_list = list(dnas.values())  # Got this by Googling how to get values from a dict

# make empty lists to count each base
a_list=[]
t_list=[]
c_list=[]
g_list=[]

seq_length = len(seq_list[0]) # Not strictly necessary, but makes the code a bit easier to read

# fill each list with as many zeroes as there are bases in the sequence
# By the end of the loop each list is [0,0,0,0...0,0]
for base_index in range(seq_length):
    a_list.append(0)
    t_list.append(0)
    c_list.append(0)
    g_list.append(0)

# go through each sequence and add tally A,T,C,G in each position
for seq in seq_list:
    for base_index in range(seq_length):
        if seq[base_index] is 'A':
            a_list[base_index] += 1    # Remember that += 1 means add one to the value.
        if seq[base_index] is 'T':
            t_list[base_index] += 1
        if seq[base_index] is 'C':
            c_list[base_index] += 1
        if seq[base_index] is 'G':
            g_list[base_index] += 1

# make the consensus sequence
consensus=""
for base_index in range(seq_length):
    comparison_group = [a_list[base_index],t_list[base_index],c_list[base_index],g_list[base_index]] # get the values at this index for all 4 lists
    maxVal = max(comparison_group) # find the most common base
    d={0:'A',1:'T',2:'C',3:'G'}    # define a dictionary to correlate position in comparison_group with a base name. This is probably awful, but it works
    consensus += d[comparison_group.index(maxVal)] # Add the base whose index is the maximum among the comparison group
    
print(consensus)
dnas = read_fasta('../input/codons/PS2Q4.fasta')

def longest_common_sequence(dna_dict):
    seq_list = list(dnas.values()) # Same as above. Get just the DNA sequences

    # Take the first sequence and divide it into all possible substrings  
    search_list = []               # this list will hold all combinations of bases that appear in the first sequence
    seq_len     = len(seq_list[1]) # length of the sequences
    first_seq   = seq_list[1]      # easier to read version of first sequence name
    
    for subset_length in range(1,seq_len+1):       # Start at 1-base strings and grow to size of whole sequence
        for pos in range(0,seq_len,subset_length): # Move the start of the subset by increasingly large steps
            search_list.append(first_seq[pos:pos+subset_length]) # Add the string to the search list
    
    # I wanted to search from big to small so I sorted my search list
    # found through Google: https://stackoverflow.com/questions/2587402/sorting-python-list-based-on-the-length-of-the-string
    search_list.sort(key=len) 
    
    for search_seq in search_list[::-1]: # Go through, starting with the biggest substring
        found_longest = True             # Just like the prime finder, assume you found it, then look for counterexamples
        for seq in seq_list:             # Loop through each of the other sequences
            if search_seq not in seq:    # If the search sequence isn't in the other sequences, then it isn't common, so mark false and break this loop
                found_longest = False
                break
        if found_longest:                # If we make it to this point with found_longest still true, then we've found our winner
            return(search_seq)

longest_common_sequence(dnas)
