import gc

import numpy as np
import pandas as pd

from tqdm import tqdm_notebook

from IPython.display import clear_output, display
from sklearn.externals.joblib import Parallel, delayed
# Get the data
train_df = pd.read_csv('../input/train.csv').set_index('ID')

# Get columns and rows in question
giba_cols = [
    "f190486d6","58e2e02e6","eeb9cd3aa","9fd594eec","6eef030c1","15ace8c9f",
    "fb0f5dbfe","58e056e12","20aa07010","024c577b9","d6bb78916",
    "b43a7cfd5","58232a6fb"
]
giba_rows = [
    '7862786dc', 'c95732596', '16a02e67a', 'ad960f947', '8adafbb52',
    'fd0c7cfc2', 'a36b78ff7', 'e42aae1b8', '0b132f2c6', '448efbb28',
    'ca98b17ca', '2e57ec99f', 'fef33cb02'
]

giba_df = train_df.loc[giba_rows, ["target"]+giba_cols]
giba_df
class GeneticOptimizer():
    
    def __init__(self, 
                 n_population=100,
                 n_breeders=10, 
                 n_lucky=2, 
                 n_generations=10, 
                 max_row_mutations=10, 
                 max_col_mutations=10, 
                 max_combined_rows=10, 
                 max_combined_cols=10):
        
        # Set variables
        self.n_population = n_population
        self.n_generations = n_generations
        self.n_breeders = n_breeders
        self.n_lucky = n_lucky
        self.max_row_mutations = max_row_mutations
        self.max_col_mutations = max_col_mutations
        self.max_combined_rows = max_combined_rows
        self.max_combined_cols = max_combined_cols
        self.history = []
        self.fittest = []
    
    @staticmethod
    def fitness(X, weights, individual):
        """
        Lower score means better alignment, see sample df at:
        https://www.kaggle.com/titericz/the-property-by-giba
        """

        # Get a copy of our dataframe       
        X = X.loc[individual['rows'], ['target','target+1'] + individual['cols'].tolist()]

        # Shift matrix to get fitness
        shiftLeftUp = X.iloc[1:, 1:].values
        deleteRightDown = X.iloc[:-1, :-1].values    

        # Calculate & return score
        score = np.sum((shiftLeftUp - deleteRightDown).astype(bool).astype(int) * weights)
        return score
    
    @staticmethod
    def hash_individual(individual):
        return hash(frozenset(individual))
    
    @staticmethod
    def swap_random(seq, n):
        """Swaps a n-length subsequence around in seq"""
        l = len(seq)
        idx = range(l)
        i1, i2 = np.random.choice(idx, 2, replace=False)
        i1 = l-n if n + i1 >= l else i1
        i2 = l-n if n + i2 >= l else i2
        for m in range(n):
            seq[i1+m], seq[i2+m] = seq[i2+m], seq[i1+m]
            
    @staticmethod
    def get_parallel(verbose=0, n_jobs=-1, pre_dispatch='2*n_jobs'):
        return Parallel(
            n_jobs=n_jobs,
            pre_dispatch=pre_dispatch,
            verbose=verbose
        )
            
    def create_initial_population(self, columns, index):
        population = []
        for _ in range(self.n_population):
            np.random.shuffle(columns)
            np.random.shuffle(index)
            population.append({'cols': np.copy(columns), 'rows': np.copy(index)})
        return np.array(population)
    
    def compute_population_performance(self, population, X, weights, **kwargs):        
        parallel = self.get_parallel(**kwargs)
        performance = parallel(
            delayed(self.fitness)(X, weights, individual) for individual in population
        )
        return np.array(performance)
    
    def select_from_population(self, population, performance, best_sample=3, lucky_few=1):
        
        # Sort the population to have best first
        sorted_population = population[np.argsort(performance)]
        
        # Save the fittest individual of the generation
        self.fittest.append(sorted_population[0])
        
        # Create next generation with best and random
        nextGeneration = []
        for i in range(best_sample):
            nextGeneration.append(sorted_population[i])
        for i in range(lucky_few):
            nextGeneration.append(np.random.choice(sorted_population))
            
        # Shuffle new generation and return
        np.random.shuffle(nextGeneration)        
        return nextGeneration
    
    def create_child(self, breeders):
        
        # Mom, dad and child
        mom = breeders[np.random.randint(0, len(breeders))]
        dad = breeders[np.random.randint(0, len(breeders))]        
        child_columns, child_index = [0]*self.n_cols, [0]*self.n_rows
        
        # Convenience function
        def set_trait(array, index, mom_trait, dad_trait):
            if np.random.rand() > 0.5:
                if mom_trait not in array:
                    array[index] = mom_trait
            else:
                if dad_trait not in array:
                    array[index] = dad_trait
        
        # Get characteristics from parent 1
        for i in range(self.n_cols):
            set_trait(child_columns, i, mom['cols'][i], dad['cols'][i])
        for i in range(self.n_rows):
            set_trait(child_index, i, mom['rows'][i], dad['rows'][i])
            
        # Fill in missing values (in a sense also a mutation factor)
        missing_cols = [c for c in mom['cols'] if c not in child_columns]
        for i in range(self.n_cols):
            if child_columns[i] == 0:
                child_columns[i] = missing_cols.pop()
                
        missing_rows = [c for c in mom['rows'] if c not in child_index]
        for i in range(self.n_rows):
            if child_index[i] == 0:
                child_index[i] = missing_rows.pop()
                
        return {'cols': np.array(child_columns), 'rows': np.array(child_index)}
    
    def create_children(self, breeders, n_children, **kwargs):
        parallel = self.get_parallel(**kwargs)
        nextPopulation = parallel(
            delayed(self.create_child)(breeders) for _ in range(n_children)
        )
        return np.array(nextPopulation)
    
    def mutate_individual(self, individual):
        if self.max_row_mutations > 0:
            for _ in np.arange(0, np.random.randint(0, self.max_row_mutations)):
                n = np.random.randint(1, self.max_combined_rows)
                self.swap_random(individual['rows'], n)
        if self.max_col_mutations > 0:
            for _ in np.arange(0, np.random.randint(0, self.max_col_mutations)):
                n = np.random.randint(1, self.max_combined_cols)
                self.swap_random(individual['cols'], n)
        return individual
    
    def mutate_population(self, population, **kwargs):
        parallel = self.get_parallel(**kwargs)
        nextPopulation = parallel(
            delayed(self.mutate_individual)(individual) for individual in population
        )
        return np.array(nextPopulation)
    
    def get_fittest_target_error(self, X, validation_index):
        """Assume first column in individual is 2 steps behind target"""
        
        individual = self.fittest[-1]
        
        target_idx = [i for i in individual['rows'] if i in validation_index]
        target = np.log1p(X.loc[target_idx, 'target'])
        
        target2p_col = individual['cols'][0]
        target2p = np.log1p(X.loc[target_idx, target2p_col].shift(-2))
        
        return np.sqrt((target-target2p).dropna()**2).sum()        
    
    def fit(self, X, y, weights=None, validation_index=None, **kwargs): 
        
        # Do not modify original
        X = X.copy()
    
        # How many rows & columns do we have
        self.n_cols = len(X.columns)
        self.n_rows = len(X.index)        
        
        # Create initial population
        population = self.create_initial_population(X.columns.tolist(), X.index.tolist())
        
        # Add target and target+1 to X, so as to direct the order of result
        X.insert(0, 'target+1', y.shift(1))
        X.insert(0, 'target', y)
        X.fillna(0, inplace=True)
        
        # If no weights specified, all columns equally important
        if weights is None:
            weights = np.ones(X.shape[1])
        
        # Run the algorithm for n_generations
        for epoch in range(self.n_generations):
            
            # Get performance for each individual in population            
            performance = self.compute_population_performance(population, X, weights, **kwargs)
            
            # Get breeders
            breeders = self.select_from_population(population, performance)
            
            # If we have a validation index, then get the train error for the best performer
            if validation_index is not None:
                train_error = self.get_fittest_target_error(X, validation_index)
            else:
                train_error = 'NaN'   
            
            # Update population
            population = self.create_children(breeders, self.n_population, **kwargs)
            
            # Mutate population before next generation
            population = self.mutate_population(population, **kwargs)            
            
            # Save to history & display
            clear_output()
            self.history.append({
                "pop_loss": np.mean(performance),
                "std_pop_loss": np.std(performance),
                "top_performer_loss": np.min(performance),
                'generation': epoch+1,
                'Train RMSLE': train_error
            })
            display(pd.DataFrame(self.history).set_index('generation'))
            
            # Just in case
            gc.collect()
# Weigh different columns differently in scoring (most important are those close to target)
weights = np.exp(-np.linspace(0, np.sqrt(giba_df.shape[1]), giba_df.shape[1]))

# Instantiate class and run on training data        
gp_opt = GeneticOptimizer(
    n_population=1000000,
    n_breeders=1000,
    n_lucky=100,
    n_generations=10,
    max_row_mutations=5,
    max_col_mutations=5,
    max_combined_rows=5,
    max_combined_cols=5
)

# Fit to data
gp_opt.fit(
    giba_df[giba_cols], giba_df['target'], 
    n_jobs=4,
    verbose=1,
    weights=weights,
    validation_index=giba_df.index.values
)
best = gp_opt.fittest[-1]
giba_df.loc[best['rows'], ['target'] + best['cols'].tolist()]
