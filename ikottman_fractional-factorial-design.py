factors = 13
import itertools as it # permutations and combinations
import string
alphabet = list(string.ascii_uppercase.replace("I", ""))

def get_num_runs(exponent):
    return 2**exponent

def get_num_basic_factors(factor):
    for i in range(0, 25):
        if (2**i > factor):
            return i

def get_interactions(basic_factors):
    interactions = []
    for i in range(2, len(basic_factors) + 1):
        interactions.extend(list(it.combinations(basic_factors, i)))
    return interactions

def compound(nonbasic_factors, interactions):
    compounds = {}
    for i, factor in enumerate(nonbasic_factors):
        compounds[factor] = interactions[-1 * (i + 1)]
    return compounds

def get_basic_values(basic_factors, num_runs):
    runs = {}
    for i, factor in enumerate(basic_factors):
        alternate = 2**(i)
        run = []
        value = 1
        while len(run) < num_runs:
            run.extend( it.repeat(value, alternate))
            value = value * -1
        runs[factor] = run
    return runs

def get_nonbasic_values(compounds, basic_values, num_runs):
    values = {}
    for factor in compounds:
        values[factor] = [] 
    for i in range(0, num_runs):        
        for factor, compound in compounds.items():
            value = 1
            for basic_factor in compound:
                value = value * basic_values[basic_factor][i]
            values[factor].append(value)
    return values

def print_runs(values, compounds):
    for factor, run in values.items():
        line = ["+" if v == 1 else "-" for v in run]
        print(factor, ' '.join(line))
    for factor, compound in compounds.items():
        print(factor, "=", ''.join(compound))

num_basic_factors = get_num_basic_factors(factors)
num_runs = get_num_runs(num_basic_factors)
basic_factors = alphabet[:num_basic_factors]
nonbasic_factors = alphabet[num_basic_factors:factors]
interactions = get_interactions(basic_factors)
compounds = compound(nonbasic_factors, interactions)
basic_values = get_basic_values(basic_factors, num_runs)
nonbasic_values = get_nonbasic_values(compounds, basic_values, num_runs)
values = dict(basic_values, **nonbasic_values)
print("Factors:", factors)
print_runs(values, compounds)
