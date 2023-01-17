### Initial Lambda Tables

### code taken from: https://www.kaggle.com/aljannico/variable-elimination



problem = {

    'K' : [{ '+k': 0.6,

            '-k': 0.4

          }],

    'KE' : [{ '+k+e': 0.4,

              '+k-e' : 0.6,

              '-k+e' : 0.8,

              '-k-e' : 0.2

           }],

    'KEL' : [{

        '+k+e+l': 0.9,

        '+k+e-l': 0.1,

        '+k-e+l': 0.6,

        '+k-e-l': 0.4,

        '-k+e+l' : 0.4,

        '-k+e-l' : 0.6,

        '-k-e+l' : 0.1,

        '-k-e-l': 0.9 

    }]

}



import copy

tables = copy.deepcopy(problem)



### Network structure

nodes = [ 'K', 'E', 'L'] # not necessary (redundant), just for clarity

net = { ('K','E'), ('K','L'), ('E','L') }

domains = {

    'K' : ['+k','-k'],

    'E' : ['+e','-e'],

    'L' : ['+l','-l']

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

    # Ok, everything is where it should be

    return True

               

                    

def eliminate(var):

    '''

    Eliminate the variable 'var' from the tables.

    '''

    schemata = [s for s in tables if var in s]

    print("Schemata with ",var,":",schemata)

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





print("Initial Lambda Tables: ",tables)            

# print("Tables after removing evidence: ",tables)            

eliminate('E')

print("Tables after removing E: ",tables)

eliminate('K')

print("Tables after removing K: ",tables)



print("\nP(L): ",normalize('L'),"\n")



tables = copy.deepcopy(problem)

remove_evidence(L='+l')

eliminate('E')

print("Tables after removing E: ",tables)

print("\nP(K|+l): ",normalize('K'),"\n")



tables = copy.deepcopy(problem)

remove_evidence(L='-l')

eliminate('E')

print("Tables after removing E: ",tables)

print("\nP(K|-l): ",normalize('K'),"\n")



tables = copy.deepcopy(problem)

remove_evidence(E='+e')

eliminate('L')

print("Tables after removing L: ",tables)

print("\nP(K|+e): ",normalize('K'),"\n")





tables = copy.deepcopy(problem)

remove_evidence(E='-e')

eliminate('K')

print("Tables after removing K: ",tables)

print("\nP(L|-e): ",normalize('L'),"\n")





tables = copy.deepcopy(problem)

remove_evidence(E='-e')

remove_evidence(L='+l')

print("Tables after removing evidence: ",tables)

print("\nP(K|-e,+l): ",normalize('K'),"\n")



# This can be solved in a more generic way, 

# but it suffices for now, as it should be pretty easy to understand



problem = {

    'K' : [{ '+k': 0.6,

            '-k': 0.4

          }],

    'KE' : [{ '+k+e': 0.4,

              '+k-e' : 0.6,

              '-k+e' : 0.8,

              '-k-e' : 0.2

           }],

    'KEL' : [{

        '+k+e+l': 0.9,

        '+k+e-l': 0.1,

        '+k-e+l': 0.6,

        '+k-e-l': 0.4,

        '-k+e+l' : 0.4,

        '-k+e-l' : 0.6,

        '-k-e+l' : 0.1,

        '-k-e-l': 0.9 

    }]

}



domains = {

    'K' : ['+k','-k'],

    'E' : ['+e','-e'],

    'L' : ['+l','-l']

}



## Computing the full joint distribution without any intelligence...

fjd = {}



print("Full Joint Distribution:")

for k in domains['K']:

    for e in domains['E']:

        for l in domains['L']:

            # Rechne mit der Netzstruktur: P(K) * P(E|K) * P(L|K,E)

            # Für jeden Knoten also die Wahrscheinlichkeitaverteilung des Knotens

            # gegeben seine Eltern (wenn keine Eltern, dann direkt der Prior)

            result = problem['K'][0][k]*problem['KE'][0][k+e]*problem['KEL'][0][k+e+l]

            fjd[k+e+l] = result # in der FJD merken

            print(k,e,l,":",problem['K'][0][k],'*',problem['KE'][0][k+e],'*',problem['KEL'][0][k+e+l],"=",result)

            

# Answer the questions



# P(L)

res = {}

res['+l'] = sum([val for key,val in fjd.items() if '+l' in key])

res['-l'] = sum([val for key,val in fjd.items() if '-l' in key])

print("\nP(L) = < +l:",res['+l'],', -l:',res['-l'],">")



# P(K|L)

res = {}

res['+k+l'] = sum([val for key,val in fjd.items() if '+l' in key and '+k' in key])

res['-k+l'] = sum([val for key,val in fjd.items() if '+l' in key and '-k' in key])

psum = res['+k+l'] + res['-k+l']

res['+k+l'] /= psum

res['-k+l'] /= psum



res['+k-l'] = sum([val for key,val in fjd.items() if '-l' in key and '+k' in key])

res['-k-l'] = sum([val for key,val in fjd.items() if '-l' in key and '-k' in key])

psum = res['+k-l'] + res['-k-l']

res['+k-l'] /= psum

res['-k-l'] /= psum



print("\nP(K|L) = < +l+k:",res['+k+l'],', +l-k:',res['-k+l'],", -l+k:",res['+k-l'],', -l-k:',res['-k-l'],">")



# P(K|+e)

res = {}

res['+k'] = sum([val for key,val in fjd.items() if '+e' in key and '+k' in key])

res['-k'] = sum([val for key,val in fjd.items() if '+e' in key and '-k' in key])

psum = res['+k'] + res['-k']

res['+k'] /= psum

res['-k'] /= psum

print("\nP(K|+e) = < +k:",res['+k'],', -k:',res['-k'],">")



# P(L|-e)

res = {}

res['+l'] = sum([val for key,val in fjd.items() if '-e' in key and '+l' in key])

res['-l'] = sum([val for key,val in fjd.items() if '-e' in key and '-l' in key])

psum = res['+l'] + res['-l']

res['+l'] /= psum

res['-l'] /= psum

print("\nP(L|-e) = < +l:",res['+l'],', -l:',res['-l'],">")



# P(-k|-e,+l)

res = {}

res['+k'] = sum([val for key,val in fjd.items() if '-e' in key and '+l' in key and '+k' in key])

res['-k'] = sum([val for key,val in fjd.items() if '-e' in key and '+l' in key and '-k' in key])

psum = res['+k'] + res['-k']

res['+k'] /= psum

res['-k'] /= psum

print("\nP(-k|-e,+l) = ",res['-k'])