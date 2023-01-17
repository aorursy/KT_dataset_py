### Initial Lambda Tables



problem = {

    'B' : [{ '+b': 0.5,

            '-b': 0.5

          }],

    'BR' : [{ '+r+b': 0.3,

             '-r+b' : 0.7,

             '+r-b' : 0.8,

             '-r-b' : 0.2

           }],

    'AB' : [{ '+a+b': 0.1,

             '-a+b' : 0.9,

             '+a-b' : 0.5,

             '-a-b' : 0.5

           }],

    'ARP' : [{

        '+p+a+r': 0.0,

        '-p+a+r': 1.0,

        '+p-a+r': 0.8,

        '-p-a+r': 0.2,

        '+p+a-r' : 0.6,

        '-p+a-r' : 0.4,

        '+p-a-r' : 1.0,

        '-p-a-r': 0.0 

    }]

}



import copy

tables = copy.deepcopy(problem)



### Network structure

nodes = [ 'B', 'R', 'A', 'P'] # not necessary (redundant), just for clarity

net = { ('B','R'), ('B','A'), ('R','P'), ('A','P') }

domains = {

    'B' : ['+b','-b'],

    'R' : ['+r','-r'],

    'A' : ['+a','-a'],

    'P' : ['+p','-p']

}



### Remove Evidence

def remove_evidence(**kwargs):

    '''

    Give evidence as values for Variables, named as the nodes

    '''

    if kwargs is not None:

        for var, value in kwargs.items():

            # Remove from tables

            print("Keep %s == %s" %(var,value))

            to_del = []

            to_add = []

            for schema in tables:

                if var in schema:

                    new_schema = schema.replace(var,'')

                    if new_schema != '': 

                        # We can have a list of dictionaries here

                        for d in tables[schema]:

                            # Form a new schema

                            new_table = {}

                            for e in d:

                                if value in e:

                                    new_e = e.replace(value,'')

                                    new_table[new_e] = d[e]

                            to_add.append((new_schema,new_table))

                    to_del.append(schema)

            # Delete obsolete tables

            for s in to_del:

                del(tables[s])

            # Add new tables

            for s,t in to_add:

                if s in tables:

                    tables[s].append(t)

                else:

                    tables[s] = [t]



                    

def combinations(schema):

    '''

    schema is given as a set. We want all possible combinations of 

    values for all variables in the set. 

    '''

    var = schema[0]

    if len(schema) == 1:

        return domains[var]

    result = []

    combs = combinations(schema[1:])

    for c in combs:

        for val in domains[var]:

            result.append(c+val)

    return result



print("Combinations for A,R,P: ",combinations('ARP'))



def matches(e,c,val):

    '''

    USAGE: if matches(e,c,val):

    '''

    # print("Match: ",e,c,val,val in e)

    if not val in e:

        # Value of the variable to eliminate does not match

        return False

    # now we have to check, if the entry e fits to the combination we want 

    # compute: first split into values

    values = [e[i:i+2] for i in range(0,len(e),2)]

    # print("  e vals: ",values)

    for v in values:

        # we already checked if the value of 

        # the variable to remove is in e   

        if v != val: 

            # Now we check if the value is in the combination we are looking for

            if not v in c:

                return False

    # Ok, everything is where is should be

    return True

               

                    

def eliminate(var):

    '''

    Eliminate the variable 'var' from the tables.

    '''

    schemata = [s for s in tables if var in s]

    print("Schemate with ",var,":",schemata)

    # new_schema = ''.join(list(set([v for s in schemata for v in s if v != var])))

    new_schema = list(set([v for s in schemata for v in s if v != var]))

    print("New schema: ",new_schema) # a list of one-character var names

    # now, populate the new schema

    new_schema_name = ''.join(new_schema)

    new_table = {}

    # determine all possible combinations

    for c in combinations(new_schema):

        value = {}

        # iterate over all values of var        

        sum_var = 0.0

        for val in domains[var]:

            value[val] = 1.0 # multiply        

            for s in schemata:

                # print("Look into schema ",s)

                for t in tables[s]:

                    for e in t:

                        if matches(e,c,val):

                            # print("  Val for ",e,c,val,":",t[e])

                            value[val] *= t[e]

            sum_var += value[val]

        # print(value) # Just show what you do

        new_table[c] = sum_var

    # Delete now obsolete schemata

    for s in schemata:

        del(tables[s])

    # Store new schema or new table for (old) schema

    if new_schema_name in tables:

        tables[new_schema_name].append(new_table)

    else:

        tables[new_schema_name] = [new_table]

            

print("Initial Lambda Tables: ",tables)            

remove_evidence(B='+b')

print("Tables after removing evidence: ",tables)            

eliminate('A')

print("Tables after removing A: ",tables)

eliminate('R')

print("Tables after removing R: ",tables)

tables = copy.deepcopy(problem)



print("Initial Lambda Tables: ",tables)            

remove_evidence(R='+r')

print("Tables after removing evidence: ",tables)            

eliminate('B')

print("Tables after removing B: ",tables)

eliminate('A')

print("Tables after removing A: ",tables)
## Now, let's normalize the final table(s)

def normalize(s):

    # first, if we have multiple tables for 

    # the schema s, we multiply the matching 

    # entries "together"

    value = {}

    for val in combinations(s):        

        value[val] = 1.0 # multiply    

        for t in tables[s]:

            for e in t:

                if e == val:

                    value[val] *= t[e]

    

    # Now, we compute the sum  over the entries

    val_sum = 0

    for val in value:

        val_sum += value[val]

    # ... and, finally, compute the ratios

    for val in value:

        value[val] /= val_sum   

    # Store new table for (old) schema

    tables[s] = [value] # keep it for whatever purpose you have in mind

    return value # ... and return the resulting table



normalize('P')
# Now, let's solve a final exercise in the same network

tables = copy.deepcopy(problem)



print("Initial Lambda Tables: ",tables)            

remove_evidence(P='-p')

print("Tables after removing evidence -p: ",tables)            

eliminate('A')

print("Tables after removing A: ",tables)

eliminate('R')

print("Tables after removing R: ",tables)

normalize('B')
problem = {

    'A' : [{ '+a': 0.4,

             '-a': 0.6

    }],

    'C' : [{ '+c': 0.9,

             '-c': 0.1

    }],

    'ACB' : [{

        '+a+c+b': 0.2,

        '+a+c-b': 0.8,

        '+a-c+b': 0.9,

        '+a-c-b': 0.1,

        '-a+c+b': 0.4,

        '-a+c-b': 0.6,

        '-a-c+b': 0.3,

        '-a-c-b': 0.7 

    }],

    'BD' : [{

        '+b+d' : 0.75,

        '+b-d' : 0.25,

        '-b+d' : 0.5,

        '-b-d' : 0.5

    }]

}



### Network structure

nodes = [ 'A', 'B', 'C', 'D'] 

net = { ('A','B'), ('B','D'), ('C','B') }

domains = {

    'A' : ['+a','-a'],

    'B' : ['+b','-b'],

    'C' : ['+c','-c'],

    'D' : ['+d','-d']

}

tables = copy.deepcopy(problem)

print("Causal question: P(D|-a+c):")

print("  Initial Lambda Tables: ",tables)            

remove_evidence(A='-a')

print("  Tables after removing evidence -a: ",tables)            

remove_evidence(C='+c')

print("  Tables after removing evidence +c: ",tables)            

eliminate('B')

print("  Tables after removing B: ",tables)

normalize('D')

print("  Result after normalization: ",tables)

tables = copy.deepcopy(problem)

print("Diagnostic question: P(A|-d), removing C before B")

print("  Initial Lambda Tables: ",tables)            

remove_evidence(D='-d')

print("  Tables after removing evidence -d: ",tables)            

eliminate('C')

print("  Tables after removing C: ",tables)

eliminate('B')

print("  Tables after removing B: ",tables)

normalize('A')

print("  Result after normalization: ",tables)
tables = copy.deepcopy(problem)

print("Diagnostic question: P(A|-d), removing B before C")

print("  Initial Lambda Tables: ",tables)            

remove_evidence(D='-d')

print("  Tables after removing evidence -d: ",tables)            

eliminate('B')

print("  Tables after removing B: ",tables)

eliminate('C')

print("  Tables after removing C: ",tables)

normalize('A')

print("  Result after normalization: ",tables)