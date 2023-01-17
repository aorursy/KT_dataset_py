import collections

import itertools as it

import logging

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

import sys

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import r2_score, mean_squared_error



%matplotlib inline
class MultiDataFrame(object):

    """

    Manages multiple DataFrames



    Loads multiple DataFrames from different csv files. Stores each 

    DataFrame in a dictionary entry.



    Attributes

    ----------

    dfs: dictionary

        The dictionary that stores each DataFrame. Changes made by the 

        methods of this class will modify these DataFrames.

    originals: dictionary

        The cache dictionary storing each unchanged DataFrame.

        

    Parameters

    ----------

    configs: dictionary of dictionaries, optional

        If no configuration is informed, the DataFrames must be further 

        loaded using the method 'add_df'.

        Each entry is formed by the pair 'df_id' and a dictionary.

        The dictionary holds the properties used to load a DataFrame from 

        a csv file and stores in dfs, using 'df_id' as key. The allowed properties are:

        'header': int, list of int, ‘infer’, default 0

            indicates each rows of csv are used as headers.

        'encoding':str, default 'utf-8'

            Encoding to use for UTF when reading/writing (ex. ‘utf-8’).

            `List of Python standard encodings

    <https://docs.python.org/3/library/codecs.html#standard-encodings>`_ .

        'file': str, optional

            csv file to load the DataFrame. If not informed,

            'df_id' + '.csv' is used instead.

    cache_original: boolean, default True

        Indicates if a copy of each DataFrame must be cached in originals. 

        This copy can be used to restore the original DataFrame into dfs. 

        If the DataFrame is not cached and the method `reload` is called, 

        the DataFrame will be restored from the csv file.



    Examples

    --------

    >>> mdf = MultiDataFrame({2016 : {}})

    

    Will load the csv file '2016.cvs', storing the DataFrame in `dfs[2016]`, 

    caching a copy into `originals[2016]`, using 'utf-8' and the first row 

    as header.



    >>> mdf = MultiDataFrame({2017 : {'header' : [0,1], 

    ...                               'encoding' : 'iso8859_2',

    ...                               'file' : 'data.csv'}})



    Will load the csv file 'data.csv', storing the DataFrame in `dfs[2017]`,

    caching a copy into `originals[2017]`, using the encoding 'iso8859_2' and the

    first and the second file rows as headers.

    

    """

    

    def _replace_Nones(self, df_id, header, encoding, file):

        """Change None values by default values.

        

        Returns

        -------

        header: [0] if it was None or same value otherwise

        encoding: 'utf-8' if it was None or same value otherwise

        file: str(df_id)+'.csv' if it was None or same value otherwise

        """

        

        header = [0] if header is None else header

        encoding = 'utf-8' if encoding is None else encoding

        file = '../input/{}.csv'.format(df_id) if file is None else file

        

        return header, encoding, file





    def __init__(self, configs={}, cache_original=True):

        

        self._cache_original = cache_original

        self._cfgs = {}

        self.dfs = {}

        self.originals = {}

        

        for df_id, cfg in configs.items():

            

            header = cfg.get('header')

            encoding = cfg.get('encoding')

            file = cfg.get('file')

            

            header, encoding, file = self._replace_Nones(df_id, header, encoding, file)



            self._cfgs[df_id] = {'header':header, 'encoding':encoding, 'file':file}

        

            # Read the cvs file and load the DataFrames

            self.dfs[df_id] = pd.read_csv(file, header=header, encoding=encoding, engine='python')

            if self._cache_original:

                self.originals[df_id] = self.dfs[df_id].copy()

            



    def add_df(self, df_id, header=None, encoding=None, file=None, cache_original=True):

        """Loads a DataFrame from csv file and stores it in `dfs[df_id]`.

        

        This method can be used to add a new DataFrame or as an alternative 

        to loading the DataFrames in the constructor.



        Parameters

        ----------

        df_id

            The id used as key to store the DataFrame in dfs

        encoding: str, default 'utf-8'

            Encoding to use for UTF when reading/writing (ex. ‘utf-8’).

            `List of Python standard encodings

    <https://docs.python.org/3/library/codecs.html#standard-encodings>`_.

        file

            File used to read the csv file and load the DataFrame.

        cache_original: bool, default True

            Determines if the DataFrame must have a copy saved in 

            'originals[df_id]'. When True, the call to `reload` method 

            makes a copy from originals to dfs, otherwise the DataFrame

            is read from csv.



        Returns

        -------

        The DataFrame loaded.

        

        See Also

        --------

        reload: Restores the original state of the DataFrame into `dfs`

        

        Warns

        -----

        When another DataFrame already existis in `dfs[df_id]` a warning

        message will informe that the old DataFrame is overridden.

        """

        

        self._chache_original = cache_original

        

        header, encoding, file = self._replace_Nones(df_id, header, encoding, file)

        

        if not self.dfs.get(df_id) is None: logging.warning("Overriding data frame with id {}".format(str(df_id)))

        

        self._cfgs[df_id] = {'header':header, 'encoding':encoding, 'file':file}

        self.dfs[df_id] = pd.read_csv(file, header=header, encoding=encoding, engine='python')

        

        if self._cache_original:

            self.originals[df_id] = self.dfs[df_id].copy()

        return self.dfs[df_id]

    

    

    def rename_columns(self, df_id, rename_dict):

        """Apply the `DataFrame.raname` to change the column names.

        

        Returns

        -------

        The DataFrame with new column names.

        """

        self.dfs[df_id] = self.dfs[df_id].rename(columns=rename_dict)

        return self.dfs[df_id]

    

    def reload(self, df_id, refresh_cache=False):

        """Restores the original state of the DataFrame into `dfs`.



        If MultiDataFrame instance was created with cache_original False,

        loads the DataFrame from csv file using the same properties used 

        to add the DataFrame at the first time. Otherwise, uses the cache 

        `originals` to restore the original state of the DataFrame.

        

        Parameters

        ----------

        df_id

            The key used to identify the DataFrame in `dfs` and recover the 

            properties used to load the csv file if needed.

        refresh_cache: bool, optional. Default False.

            Indicates if both `originals[df_id]` and `dfs[df_id]` must be

            reloaded by reading the original DataFrame from csv file. If 

            `cache_original` is False this parameter is ignored and the 

            file is always read.



        Returns

        -------

        The DataFrame as it is from cvs file.

        """

        if (not self._cache_original) or refresh_cache:

            

            cfg = self._cfgs[df_id]

            df = pd.read_csv(str(cfg.file), header=cfg.header, encoding=cfg.encoding, engine='python')

            self.dfs[df_id] = df

            

            if refresh_cache:

                self.originals[df_id] = df.copy()

            

        else:

            self.dfs[df_id] = self.originals[df_id].copy()

        

        return self.dfs[df_id]



    def set_column_multiIndex(self, df_id, column_names):

        """Create a MultiIndex column from the column names informed.

        

        Uses the new MultiIndex to update the column indexes of the 

        DataFrame located at `dfs[df_id]`.

        

        Parameters

        ----------

        df_id

            The key to identify the DataFrame in `dfs` that will have the 

            MultiIndex columns.

        column_names: list or sequence of array-likes

            Each array-like gives one level’s name for each column. 

            len(column_names) is the number of levels.

        

        Returns

        -------

        The DataFrame updated.

        """

        self.dfs[df_id].columns = pd.MultiIndex.from_arrays(column_names)

        return self.dfs[df_id]

    

    def drop_column(self, df_id, columns, level=None):

        """Drop the columns from the DataFrame `dfs[df_id]`.

        

        The operation is executed inplace in `dfs[df_id]`.

        

        Parameters

        ----------

        df_id

            The key used to identify the DataFrame in `dfs` that will have

            the columns dropped.

        columns: str or list of column names

            The column or columns that will be dropped from the DataFrame.

        

        Returns

        -------

        The DataFrame with the columns removed.

        

        Raises

        ------

        AttributeError

            - If the parameter `columns` is not a str to identify a single 

            column or an Iterable to identify multiple columns.

            - If the columns use MultiIndex and the level wasn't specified.

        

        Warns

        -----

            - When the column is not found in the column's index.

            - When the columns don't use MultiIndex and a level id 

            specified.

            

        See Also

        --------

        `DataFrame.drop

        <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html#pandas.DataFrame.drop>_`

        """

        

        if type(columns) is str:

            # String is also iterable, but must be considered as a single column

            columns = [columns]

            

        elif not isinstance(columns, collections.abc.Iterable):

            #Check if it is an Iterable for multiple column names

            

            msg = "Unknown type {}. It should be a string for single column or an Iterable otherwise."

            raise AttributeError(msg.format(type(columns)) )

        

        assert isinstance(columns, collections.abc.Iterable)

        

        #Drop the columns in the list of column names

            

        df = self.dfs[df_id]



        if isinstance(df.columns, pd.MultiIndex):

            if level is None:

                raise AttributeError("Level must be defined when DataFrame uses MultiIndex.")



            for c in columns:

                # When level is not None and the key is not found, the

                # method `DataFrame.drop` doesn't raise Exception

                #Check if the columns exists to know if it can be dropped

                if c in df.columns.to_frame()[level].values:

                    df.drop(columns=c, level=level, inplace=True)

                else:

                    logging.warning("Key '{}' not found in level {} of DataFrame {}.".format(c, level, df_id))

                    

        else:

            if not level is None:

                # The level is useless in a DataFrame without MultiIndex

                logging.warning("Level is ignored when the DataFrame does not use MultiIndex.")



            for c in columns:

                try:

                    self.dfs[df_id].drop(columns=c, inplace=True)



                except KeyError:

                    # If the key doesn't exist, catch the exception and log a message

                    logging.warning("Key '{}' not found in level {} of DataFrame {}.".format(c, level, df_id))

        

        return self.dfs[df_id]

    

    def value_counts(self, df_id, column, normalize=True, dropna=False):

        """ Counts unique values in the DataFrame and columns specified.



        Can count unique values in multiple Series, as long it is in the 

        same DataFrame. The counting will be in descending order, being the 

        mostfrequently-occurring elements at the top. By default NA values 

        are included and the countings are relative frequencies.



        Parameters

        ----------

        df_id

            The key used to identify the DataFrame in which the columns 

            will be retrieved to execute the counting.

        column: str or Iterable

            Str for single column or a list of column names in which the 

            counting will be executed.

        normalize: bool, default True

            Whether or not the counting must be relative frequencies.

        dropna: bool, default False

            Whether or not the NA values must be included in countings.

        """

        df = self.dfs[df_id][column]



        if type(df) is pd.Series:

            

            counts = df.value_counts(normalize=normalize, dropna=dropna)

            

            return pd.DataFrame(counts)



        elif type(df) is pd.DataFrame:



            main_df = pd.DataFrame(columns=pd.MultiIndex(levels=[[],[]], codes=[[], []]))



            for c in df.columns:



                counts = df[c].value_counts(normalize=normalize, dropna=dropna)

                label_counts = "Counts %" if normalize else "Counts"

                df_counts = pd.DataFrame({(c, "Items"):counts.index.values, (c, label_counts):counts.values})



                main_df = main_df.merge(df_counts, right_index=True, left_index=True, how='outer')



            return main_df

        

    def get_column_names(self, df_id, level=None):

        """Retrieves the names of the columns.



        When the column index is MultiIndex only retrieves the names from a

        single level. In such cases, the level must be specified. If the 

        column index is not a MultiIndex, ignores the level and registers 

        a warning message.

        

        Parameters

        ----------

        df_id

            The key used to identify the DataFrame in which the columns

            will be retrieved.

        level: int

            The level from where the column names will be retrieved. Only 

            useful if the column index is MultiIndex, it is ignored 

            otherwise.

            

        Returns

        -------

        A list with the name of the columns.

            

        Raises

        ------

        AttributeError

            - If the columns use MultiIndex and the level wasn't informed.

            

        Warns

        -----

            - When the parameter `level` is informed but the columns don't

            use MultiIndex.

        

        """

        df = self.dfs[df_id]

        

        if type(df.columns) is pd.Index:

            if not level is None:

                logging.warning("The attribute level is ignored when the columns do not use MultiIndex.")

            

            return df.columns.values.tolist()

        

        elif isinstance(df.columns, pd.MultiIndex):

            if level is None:

                raise AttributeError("The level must be specified when the columns are MultiIndex.")

                

            return [col[level] for col in df.columns]

            

        else:

            msg = "Unknown type of index for columns: {}.".format(type(df.columns))

            raise AttributeError(msg)

        

    def print_df_column_names(self, df_id):

        """Prints the column names. Prints names from all levels if 

        columns use MultiIndex

        

        Print each column in a row. If the column use MultiIndex, the 

        pattern {first column - level 0} >> {first column - level 1} >> ...

        is used.

        

        Parameters

        ----------

        df_id

            The id used to identify the DataFrame from where the column 

            names are retrieved.

        """

        print_patter = "{:<23}"



        df = self.dfs[df_id]

        

        if type(df.columns) is pd.MultiIndex:

            print_patter = [print_patter] * len(df.columns[0])

            print_patter = " >> ".join(print_patter)

            for names in df.columns.values:

                print(print_patter.format(*names))

        else:

            for names in df.columns.values:

                print(print_patter.format(names))

                

    def print_empty_rename_map(self, df_id, level=0):

        """Prints a map used to rename the column names.

        

        The map is partially filled with the keys as the original column 

        names and a empty str where the new name must be placed.



        Parameters

        ----------

        df_id

            The id used to identify the DataFrame from where the column 

            names are retrieved.

        level: int, default 0

            The level of column index that will be renamed and should be 

            used to build the empty map.

        """

        df = self.dfs[df_id]

        template = "'''{}''' : , str()"

        if isinstance(df.columns, pd.MultiIndex):

            for col in df.columns.values:

                print(template.format(col[level]))

        else:

            if level > 0:

                msg = "Ignoring parameter level={} for not MultiIndex columns.".format(level)

                logging.warning(msg)

            

            for col in df.columns.values:

                print(template.format(col))



    def __repr__(self):

        """Build the representation of MultiDataFrame.

        

        Shows the configuration used to read the csv files and the shapes 

        of all the DataFrames loaded.

        """

        csv_st = "Configuration\n"

        df_st = "DataFrames\n"

        

        template_csv = '{} - encoding: {encoding:10s}; headers: {header}; file: {file}\n'

        template_df = '{0} - rows: {1:>5}; columns: {2:>5}\n'

        

        for df_id, cfg in self._cfgs.items():

            

            df = self.dfs[df_id]

            csv_st +=  template_csv.format(str(df_id), **cfg)

            df_st += template_df.format(str(df_id), *df.shape)

            

        return csv_st + df_st

            

    def find_occurrences(self, wanted_terms, df_ids=None):

        """Execute a search over DataFrames to find the occurrences of 

        the `wanted_terms`.

        

        The match is done ignoring case and uses str representation of 

        each item in `wanted_terms`. If no df_id is informed, the search 

        is executed over all DataFrames.

        

        Parameters

        ----------

        wanted_terms: a str term or iterable of terms

            Each term is converted to str to used in the search.

        df_ids: list of keys to identify DataFrames in dfs, optional

            If no key is informed the searches in all DataFrames in `dfs`

        

        Returns

        -------

        This is a map containing the occurrences. The key of the map is the

        same key of the DataFrame in `dfs`. The value of the map consists 

        of another map, in which the keys are the columns and the values 

        are the occurrences found.

        """

        items_found = {}

        

        if type(wanted_terms) is str:

            # If a single str was informed, transforms it into a list

            wanted_terms = [wanted_terms]

            

        if df_ids is None:

            # If no df_id was informed, search over all DataFrames

            df_ids = self.dfs.keys()

        

        def find_in_series(item, bag):

            # Method to search only over pd.Series

            for term in wanted_terms:

                if term.casefold() in str(item).casefold():

                    bag.add(str(item))

                    return;

        

        for df_id in df_ids:

            # Execute the search over the each DataFrame

            

            items_found[df_id] = {}

            #Creates the entry for the current DataFrame

            

            df = self.dfs[df_id]

            

            for col in df.columns:

                #Execute the search over each column



                bag = set()#No repetitions allowed

                df[col].apply(find_in_series, args=(bag,))

                if len(bag) > 0:

                    items_found[df_id][col] = bag

        return items_found

        

              

    def count_occurrences(self, wanted_terms, df_ids=None, verbose=True):

        """Count the occurrences of the `wanted_terms`.



        Counts the total occurrences of any `wanted_terms` in each 

        DataFrame. If the word 'dog' occurs three times in the DataFrame 

        and the word 'cat' occurs two times, the result will be five for 

        the `wanted_terms=['dog', 'cat']`.

        

        Parameters

        ----------

        wanted_terms: str term or list of terms

            The set of terms used to count the occurrences.

        df_ids: key or list of keys o DataFrames

            Defines in which DataFrame the occurrences must be count. If 

            no df_id is informed, all DataFrames will be considered.

        verbose: bool, default True

            Determines if it should print how many occurrrences happen in 

            each DataFrame.

            

        Returns

        -------

        A map with the DataFrame id as key and the total occurrences of 

        the wanted_terms happened in the respective DataFrame.

        """

        if type(wanted_terms) is str:

            wanted_terms = [wanted_terms]

            

        if df_ids is None:

            df_ids = self.dfs.keys()

        

        def term_counter(item):

            #This function is called for each collumn of the DataFrame

            #Returns the sum all the occurrences of the items in wanted_terms

            return sum([term in str(item).casefold() for term in wanted_terms])

        

        counts = {}

        

        for df_id in df_ids:

            #Do the count to each DataFrame

            df = self.dfs[df_id]

            

            #Apply the term_counter function in each column

            count = df.applymap(term_counter).sum().sum()

            #Stores the sum over each column and over all columns

            

            counts[df_id] = count

            if verbose: print("Dataset: {}, occurrences: {:>6}".format(df_id, count))

            

        return counts

    

    def show_shared_columns(self, df_ids=None):

        """Shows which columns is present in each DataFrame.

        

        Builds a new DataFrame using all the columns of all DataFrames in 

        dfs as row index and use the keys of the DataFrams in dfs as 

        column index.

        

        Parameters

        ----------

        df_ids: list of keys

            The keys of the DataFrames in `dfs` to look for shared columns.

            

        Returns

        -------

        A DataFrame describing what columns are shared between multiple 

        DataFrames.

        

        Examples

        --------

        >>> mdf.show_shared_columns([2011, 2017])

                              2011    2017    count

        Years Programming        1       1        2

        Country                  1       1        2

        Occupation               1       1        2

        Company Size             1       1        2

        Technology               1       1        2

        ...     ...     ...     ...

        Important Hiring Rep     0       1        1

        Important Hiring TechExp 0       1        1

        Important Hiring Titles  0       1        1

        Important Job Security   0       1        1

        origin                   1       0        1

        """

        

        if df_ids is None:

            df_ids = self.dfs.keys()

        

        df_columns = pd.DataFrame()

        

        for df_id in df_ids:

            col_names = list({item[0] for item in self.dfs[df_id].columns})

            values = np.ones(len(col_names))

            s = pd.Series(values, index=col_names, name=df_id).sort_values()

            df_columns = df_columns.merge(s, how='outer', left_index=True, right_index=True)

            

        df_columns = df_columns.fillna(0)

        df_columns = df_columns.astype('int32')

        df_columns = df_columns.assign(count = lambda c : c.sum(axis=1))

        df_columns = df_columns.sort_values('count', ascending=False)

        

        return df_columns

    

    def find_common_columns(self, df_ids=None):

        """Find the commom columns for each possible combination of 

        DataFrames.

        

        Find the common columns in each combination possible of DataFrames,

        ranging the group size from two by two to all DataFrames.

        

        Parameters

        ----------

        df_ids: list of keys with at least two keys

        

        Examples

        --------

        >>> mdf.find_common_columns([2011, 2015, 2019])

    columns                                        combs              count

2   {StackO Visited Sites, Occupation, Gender...   (2015, 2019)       17.0

1   {StackO Visited Sites, Occupation, Purcha...   (2011, 2019)       11.0

0   {StackO Visited Sites, Occupation, Indust...   (2011, 2015)       10.0

3   {StackO Visited Sites, Occupation, Purcha...   (2011, 2015, 2019)  9.0

        """

        

        if df_ids is None:

            df_ids = self.dfs.keys()



        def get_columns_intersection(df_ids):

            assert(len(df_ids) >= 2)

            inter_cols = {col[0] for col in self.dfs[df_ids[0]].columns}



            for df_id in df_ids[1:]:

                inter_cols = inter_cols.intersection({item[0] for item in self.dfs[df_id].columns})

            return inter_cols



        possible_combinations = []

        for comb_len in range(2, len(df_ids)+1):

            n_combs = list(it.combinations(df_ids, comb_len))

            for comb in n_combs:

                possible_combinations.append(comb)



        common_columns = []

        for comb in possible_combinations:

            common_columns.append(get_columns_intersection(comb))



        frame = pd.DataFrame()

        for i in range(len(possible_combinations)):

            frame = frame.append(pd.Series({'count':len(common_columns[i]),

                                            'columns':common_columns[i],

                                            'combs':possible_combinations[i]}), ignore_index=True)

        

        frame = frame.sort_values('count', ascending=False)

        return frame

    

    def append(self, df_ids=None, keep_result=True, add_id_columns=True, sort=False, ignore_index=True, verbose=True):

        """Append all DataFrames in dfs in a new DataFrame.

        

        Appends all rows of the DataFrames in `dfs`, resulting in a new 

        DataFrame. Reuses the columns that appear in more than one 

        DataFrame, appending values at the bottom.  The missing columns 

        are filled with NaNs. The final DataFrame will have as many columns

        as the union of all the columns from all DataFrames in `dfs`.

        

        Stores the resulting DataFrame into `self.appended`.

        

        Parameters

        ----------

        df_ids: list of keys of DataFrames

            A list describing the DataFrames that should me appended 

            together.

        keep_result: bool, default True

            Whether it should keep the resulting DataFrame in the attribute

            `mdf.appended`.

        add_id_columns: bool, default True

            Whether it should include a new column in the resulting 

            DataFrame to identify from which DataFrame the rows came 

            from. When it is True, uses the DataFrame key in `dfs` as 

            identifier.

        sort: bool, default False

            Sort columns if the columns of the DataFrames are not aligned.

        ignore_index: bool, default True

            If True, does not use the index labels.

            

        See Also

        --------

        `pd.DataFrame.append 

        <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.append.html#pandas.DataFrame.append>`

        """

        if df_ids is None:

            df_ids = list(self.dfs.keys())

        

        if len(df_ids) < 2:

            msg = "It is not possible to execute merge with {} DataFrame. The minimum is 2.".format(len(df_ids))

            raise AttributeError(msg)

        

        result_append = self.dfs[df_ids[0]].copy()# Initialize with the first DataFrame

        

        if add_id_columns:

            first_df_id = df_ids[0]

            result_append['origin'] = [first_df_id] * len(result_append)

            if verbose: print("Dataset {} appended".format(first_df_id))

        

        for df_id in df_ids[1:]:

            current_df = self.dfs[df_id].copy()

            if add_id_columns:

                current_df['origin'] = [df_id] * len(current_df)

            result_append = result_append.append(current_df, sort=sort, ignore_index=ignore_index)

            if verbose: print("Dataset {} appended".format(df_id))

        

        if keep_result: self.appended = result_append

        

        return result_append

    

    def unique_values(self, df_id=None, columns=None, keep_result=True):

        """Returns the unique values of each column in the DataFrame.

        

        Results in a DataFrame with the same columns as specified in 

        `columns`. Each column holds the list of unique items. When a 

        column has less unique items then the other, the list is filled 

        with NaNs.

        

        Parameters

        ----------

        df_id

            The id used to identify the DataFrame with the columns to 

            retrieve the unique values.

        columns: str or Iterable, for single or multiple columns. Optional.

            Uses all the columns when this parameters is not provided.

        keep_result: bool, default True.

            Whether it should store the result in `mdf.uniques`

        

        Returns

        -------

        A DataFrame with columns' unique values.

        """

        

        if df_id is None:

            df = self.appended

        else:

            df = self.dfs[df_id]

            

        if columns is None:

            df_columns = df.columns

        elif type(columns) is str:

            df_columns = df[[columns]].columns

        else:

            df_columns = df[columns].columns



        sys.stdout.write("\r[%-20s] %d%%" % ('='*0, 0))

        

        if type(df_columns) is pd.MultiIndex:

            unique_values = pd.DataFrame(columns=[[] * df_columns.nlevels])

        else:

            unique_values = pd.DataFrame(columns=[[]])

        

        total_cols = len(df.columns)

        

        progress=0

        for column in df_columns:

            

            sx = pd.Series(df[column].unique(), name=column)

            unique_values = unique_values.merge(sx, how='outer', right_index=True, left_index=True)

            

            sys.stdout.write("\r[%-20s] %d%%" % ('='*(progress*20//total_cols), progress*100//total_cols))

            sys.stdout.flush()

            progress += 1



        sys.stdout.write("\r[%-20s] %d%%" % ('='*(20), 100))

        if keep_result : self.uniques = unique_values

        return unique_values

    

    def expand_categories(self, column, df_id=None, sep=";", template_name="{0}_{1}", replace=False):

        """Creates dummy variables for multiple values stored in a single 

        column.

        

        Creates one dummy variable for each value stored in a column. Uses 

        a separator to distinguish between multiple values in the column.

        

        Parameters

        ----------

        column: Column identifier

            Identifies the column in the DataFrame containing the values.

        df_id

            The key of the DataFrame in `dfs`.

        sep: str

            The separator used to split multiple values in stored in a 

            given row of the column.

        template_name: str, default '{0}_{1}'

            A tamplete to create the columns for the dummy variables. The 

            value in `0` is the original column name and the value in `1` 

            is a single value obtained from the split.

        replace: bool, default False

            Whether it should replace the original column in the DataFrame 

            with the new dummy variables.

        

        Returns

        -------

        A DataFrame with the new dummy variables.

        """

        if df_id is None:

            df = self.appended

        else:

            df = self.dfs[df_id]



        feat = df[column]



        sys.stdout.write("\r[%-20s] %d%%" % ('='*0, 0))



        unique_values = set()

        for i in feat:

            if type(i) == str:

                vals = i.split(sep)

                for v in vals:

                    unique_values.add(v.strip())



        def contains(target, term):

            return int(term.casefold() in str(target).casefold())



        progress=0

        total_uniques = len(unique_values)

        expanded = {}

        for uvalue in unique_values:



            if type(feat.name) is tuple:

                new_feat_name = list(feat.name)

                new_feat_name[-1] = template_name.format(feat.name[-1], uvalue)

                new_feat_name = tuple(new_feat_name)

            else:

                new_feat_name = (feat.name[0], new_feat_name)



            result = pd.Series(feat.apply(contains, args=[uvalue]), index=feat.index)

            expanded[new_feat_name] = result



            if replace:

                df[new_feat_name] = result



            sys.stdout.write("\r[%-20s] %d%%" % ('='*(progress*20//total_uniques), progress*100//total_uniques))

            sys.stdout.flush()

            progress += 1



        if replace:

            df.drop(columns=column, inplace=True)



        sys.stdout.write("\r[%-20s] %d%%" % ('='*(20), 100))

        return pd.DataFrame(expanded)

    
# Load all csv files

# All csv files were conveniently renamed to be like 'year.csv'

mdf = MultiDataFrame()

mdf.add_df(2011, header=[0,1], encoding='iso8859_2')

mdf.add_df(2012, header=[0,1], encoding='iso8859_2')

mdf.add_df(2013, header=[0,1], encoding='iso8859_2')

mdf.add_df(2014, header=[0,1], encoding='iso8859_2')

mdf.add_df(2015, header=[0,1], encoding='iso8859_2')

mdf.add_df(2016)

mdf.add_df(2017)

mdf.add_df(2018)

mdf.add_df(2019)

mdf
mdf.reload(2011)

mdf.reload(2012)

mdf.reload(2013)

mdf.reload(2014)

mdf.reload(2015)

mdf.reload(2016)

mdf.reload(2017)

mdf.reload(2018)

mdf.reload(2019)
occupation_2017 = mdf.value_counts(2017, ("DeveloperType"))

occupation_2017.columns = pd.MultiIndex.from_arrays([["2017"],["Counts %"]])



occupation_2018 = mdf.value_counts(2018, ("DevType"))

occupation_2018.columns = pd.MultiIndex.from_arrays([["2018"],["Counts %"]])



occupation_2019 = mdf.value_counts(2019, ("DevType"))

occupation_2019.columns = pd.MultiIndex.from_arrays([["2019"],["Counts %"]])



from_2017_to_2019 = occupation_2017.merge(occupation_2018, left_index=True, right_index=True, how='outer')

from_2017_to_2019 = from_2017_to_2019.merge(occupation_2019, left_index=True, right_index=True, how='outer')

from_2017_to_2019
occupation_2011 = mdf.value_counts(2011, ('Which of the following best describes your occupation?', 'Response'))

occupation_2011.columns = pd.MultiIndex.from_arrays([["2011"],["Counts %"]])



occupation_2012 = mdf.value_counts(2012, ('Which of the following best describes your occupation?','Response'))

occupation_2012.columns = pd.MultiIndex.from_arrays([["2012"],["Counts %"]])



occupation_2013 = mdf.value_counts(2013, ('Which of the following best describes your occupation?','Response'))

occupation_2013.columns = pd.MultiIndex.from_arrays([["2013"],["Counts %"]])



occupation_2014 = mdf.value_counts(2014, ('Which of the following best describes your occupation?','Response'))

occupation_2014.columns = pd.MultiIndex.from_arrays([["2014"],["Counts %"]])



occupation_2015 = mdf.value_counts(2015, ("Unnamed: 5_level_0","Occupation"))

occupation_2015.columns = pd.MultiIndex.from_arrays([["2015"],["Counts %"]])



occupation_2016 = mdf.value_counts(2016, ("occupation"))

occupation_2016.columns = pd.MultiIndex.from_arrays([["2016"],["Counts %"]])



from_2011_to_2016 = occupation_2011.merge(occupation_2012, left_index=True, right_index=True, how='outer')

from_2011_to_2016 = from_2011_to_2016.merge(occupation_2013, left_index=True, right_index=True, how='outer')

from_2011_to_2016 = from_2011_to_2016.merge(occupation_2014, left_index=True, right_index=True, how='outer')

from_2011_to_2016 = from_2011_to_2016.merge(occupation_2015, left_index=True, right_index=True, how='outer')

from_2011_to_2016 = from_2011_to_2016.merge(occupation_2016, left_index=True, right_index=True, how='outer')

from_2011_to_2016
# Answers to the question "Do you believe in aliens?".

mdf.value_counts(2016, ('Believe in Aliens', 'aliens'))
class FeatureName(object):

    """Keeps track of all names used to rename the columns of DataFrames.



    Attributes

    ----------

    instances: class attribute

        It is a map that keeps the all instances of this class. The key of 

        this map is the feature name.

    name: str

        The name of the feature. This class uses this attribute to keep 

        track of all names already used. Emits a warning to inform that 

        the previous instance is being overwritten when more than one 

        instance is created with the same name.

    ftype: 'numerical' or 'categorical'

        Possible types of the features.



    Parameters

    ----------

    name: str

        Feature name.

    ftype: 'numerical' or 'categorical'

        Feature type.

    """

    

    instances = {}

    ftypes = {'numerical', 'categorical'}

    

    def __init__(self, name, ftype):

        

        assert ftype in FeatureName.ftypes

        assert type(name) is str

        

        self.name = name

        self.ftype = ftype

        if self.instances.get(name) :

            logging.warning("overwriting previous value of '{}'.".format(self.instances[name]))

        self.instances[name] = self

        

    def __str__(self):

        return self.name;

    

    def __repr__(self):

        return self.__str__();

    

    def __add__(self, other):

        """Combines with other to generate a new instance of FeatureName.

        

        The new instance is generated with self.name + other. The new 

        instance ftype is the same of self.

        

        Parameters

        ----------

        other: str or FeatureName

            Combines the name of current instance with other str. If other 

            is FeatureName, uses other.name and ignores other.ftype.

        

        Returns

        -------

        A new instance of FeatureName

        """

        

        if type(other) is str:

            return FeatureName(self.name + other, self.ftype)

        

        elif type(other) is FeatureName:

            

            if self.ftype == other.ftype:

                return FeatureName(self.name + other.name, self.ftype)

            else: raise AttributeError("The ftype '{}' is not compatible with ftype '{}'".format(self.ftype, other.ftype))

                

        else:

            raise AttributeError

            

    def __eq__(self, other):

        """Compares with other.

        

        If other is str, compares only the name with other str. If other 

        if FeatureName, compare both names and ftypes.

        

        Parameters

        ----------

        other: str, FeatureName

            If another type is passed it will return False.

            

        Returns

        -------

        bool

        """

        

        if type(other) is str:

            return self.name == other

        

        elif type(other) is FeatureName:

            

            if (self.ftype == other.ftype) and (self.name == other.name):

                return True

        else:

            return False

    

    def __hash__(self):

        """Hash of its name."""

        return hash(self.name)

           
# Defining the feature names

feat_country = FeatureName('Country', 'categorical')

feat_us_region = FeatureName('US Region', 'categorical')

feat_age = FeatureName('Age', 'numerical')

feat_years_programming = FeatureName('Years Programming', 'numerical')

feat_industry = FeatureName('Industry', 'categorical')

feat_company_size = FeatureName('Company Size', 'numerical')

feat_occupation = FeatureName('Occupation', 'categorical')

feat_rec_influence = FeatureName('Recommendation Influence', 'numerical')

feat_purch_role = FeatureName('Purchasing Role', 'categorical')

feat_purch_target = FeatureName('Purchasing Target', 'categorical')

feat_budget = FeatureName('Budget', 'numerical')

feat_work_platform = FeatureName('Work Platform', 'categorical')

feat_tech_proficiency = FeatureName('Technology', 'categorical')

feat_os = FeatureName('OS', 'categorical')

feat_career_satisfaction = FeatureName('Career Satisfaction', 'numerical')

feat_annual_salary_plus_bonus = FeatureName('Annual Salary +Bonus (USD)', 'numerical')

feat_gadget = FeatureName('Gadget', 'categorical')

feat_money_tech = FeatureName('Money Spent on Gadgets (12-mos)', 'numerical')

feat_SO_freq_site = FeatureName('StackO Visited Sites', 'numerical')



# Creating the dictionary to rename the columns

rename_2011_dict = {'What Country or Region do you live in?': str(feat_country),

                    'Which US State or Territory do you live in?': str(feat_us_region),

                    'How old are you?': str(feat_age),

                    'How many years of IT/Programming experience do you have?': str(feat_years_programming),

                    'How would you best describe the industry you work in?': str(feat_industry),

                    'Which best describes the size of your company?': str(feat_company_size),

                    'Which of the following best describes your occupation?': str(feat_occupation),

                    'How likely is it that a recommendation you make will be acted upon?': str(feat_rec_influence),

                    'What is your involvement in purchasing? You can choose more than 1.': str(feat_purch_role),

                    'Unnamed: 9_level_0': str(feat_purch_role),

                    'Unnamed: 10_level_0': str(feat_purch_role),

                    'Unnamed: 11_level_0': str(feat_purch_role),

                    'Unnamed: 12_level_0': str(feat_purch_role),

                    'Unnamed: 13_level_0': str(feat_purch_role),

                    'Unnamed: 14_level_0': str(feat_purch_role),

                    'What types of purchases are you involved in?': str(feat_purch_target),

                    'Unnamed: 16_level_0': str(feat_purch_target),

                    'Unnamed: 17_level_0': str(feat_purch_target),

                    'Unnamed: 18_level_0': str(feat_purch_target),

                    'Unnamed: 19_level_0': str(feat_purch_target),

                    'Unnamed: 20_level_0': str(feat_purch_target),

                    'What is your budget for outside expenditures (hardware), software), consulting), etc) for 2011?': str(feat_budget),

                    'Unnamed: 22_level_0': str(feat_budget),

                    'Unnamed: 23_level_0': str(feat_budget),

                    'Unnamed: 24_level_0': str(feat_budget),

                    'Unnamed: 25_level_0': str(feat_budget),

                    'Unnamed: 26_level_0': str(feat_budget),

                    'Unnamed: 27_level_0': str(feat_budget),

                    'Unnamed: 28_level_0': str(feat_budget),

                    'What type of project are you developing?': str(feat_work_platform),

                    'Which languages are you proficient in?': str(feat_tech_proficiency),

                    'Unnamed: 31_level_0': str(feat_tech_proficiency),

                    'Unnamed: 32_level_0': str(feat_tech_proficiency),

                    'Unnamed: 33_level_0': str(feat_tech_proficiency),

                    'Unnamed: 34_level_0': str(feat_tech_proficiency),

                    'Unnamed: 35_level_0': str(feat_tech_proficiency),

                    'Unnamed: 36_level_0': str(feat_tech_proficiency),

                    'Unnamed: 37_level_0': str(feat_tech_proficiency),

                    'Unnamed: 38_level_0': str(feat_tech_proficiency),

                    'Unnamed: 39_level_0': str(feat_tech_proficiency),

                    'Unnamed: 40_level_0': str(feat_tech_proficiency),

                    'Unnamed: 41_level_0': str(feat_tech_proficiency),

                    'Unnamed: 42_level_0': str(feat_tech_proficiency),

                    'What operating system do you use the most?': str(feat_os),

                    'Please rate your job/career satisfaction': str(feat_career_satisfaction),

                    'Including bonus, what is your annual compensation in USD?': str(feat_annual_salary_plus_bonus), 

                    'Which technology products do you own? (You can choose more than one)': str(feat_gadget),

                    'Unnamed: 47_level_0': str(feat_gadget),

                    'Unnamed: 48_level_0': str(feat_gadget),

                    'Unnamed: 49_level_0': str(feat_gadget),

                    'Unnamed: 50_level_0': str(feat_gadget),

                    'Unnamed: 51_level_0': str(feat_gadget),

                    'Unnamed: 52_level_0': str(feat_gadget),

                    'Unnamed: 53_level_0': str(feat_gadget),

                    'Unnamed: 54_level_0': str(feat_gadget),

                    'Unnamed: 55_level_0': str(feat_gadget),

                    'Unnamed: 56_level_0': str(feat_gadget),

                    'Unnamed: 57_level_0': str(feat_gadget),

                    'Unnamed: 58_level_0': str(feat_gadget),

                    'Unnamed: 59_level_0': str(feat_gadget),

                    'Unnamed: 60_level_0': str(feat_gadget),

                    'Unnamed: 61_level_0': str(feat_gadget),

                    'Unnamed: 62_level_0': str(feat_gadget),

                    'In the last 12 months, how much money have you spent on personal technology-related purchases? ': str(feat_money_tech),

                    'Which of our sites do you frequent most?': str(feat_SO_freq_site)}
mdf.rename_columns(2011, rename_2011_dict)
feat_SO_carrer_aware = FeatureName('Aware of StackO careers', 'categorical')

feat_SO_carrer_profile = FeatureName('Have StackO career profile', 'categorical')

feat_SO_carrer_profile_why_not = FeatureName('Why not StackO career profile', 'categorical')

feat_SO_carrer_profile_why_not_other = FeatureName('Why not (other)', 'categorical')

feat_ad_rate = FeatureName('Ads rate', 'numerical')

feat_ads_seen = FeatureName('Advertisers seen', 'categorical')

feat_SO_reputation = FeatureName('StackO reputation', 'numerical')

feat_SO_freq_site_other = FeatureName(str(feat_SO_freq_site) + ' other', 'numerical')



rename_2012_dict = {'''What Country or Region do you live in?''' : str(feat_country),

                    '''Which US State or Territory do you live in?''' : str(feat_us_region),

                    '''How old are you?''' : str(feat_age),

                    '''How many years of IT/Programming experience do you have?''' : str(feat_years_programming),

                    '''How would you best describe the industry you currently work in?''' : str(feat_industry),

                    '''Which best describes the size of your company?''' : str(feat_company_size),

                    '''Which of the following best describes your occupation?''' : str(feat_occupation),

                    '''What is your involvement in purchasing products or services for the company you work for? (You can choose more than one)''' : str(feat_purch_role),

                    '''Unnamed: 8_level_0''' : str(feat_purch_role),

                    '''Unnamed: 9_level_0''' : str(feat_purch_role),

                    '''Unnamed: 10_level_0''' : str(feat_purch_role),

                    '''Unnamed: 11_level_0''' : str(feat_purch_role),

                    '''Unnamed: 12_level_0''' : str(feat_purch_role),

                    '''Unnamed: 13_level_0''' : str(feat_purch_role),

                    '''What types of purchases are you involved in?''' : str(feat_purch_target),

                    '''Unnamed: 15_level_0''' : str(feat_purch_target),

                    '''Unnamed: 16_level_0''' : str(feat_purch_target),

                    '''Unnamed: 17_level_0''' : str(feat_purch_target),

                    '''Unnamed: 18_level_0''' : str(feat_purch_target),

                    '''Unnamed: 19_level_0''' : str(feat_purch_target),

                    '''What is your budget for outside expenditures (hardware), software), consulting), etc) for 2011?''' : str(feat_budget),

                    '''What type of project are you developing?''' : str(feat_work_platform),

                    '''Which languages are you proficient in?''' : str(feat_tech_proficiency),

                    '''Unnamed: 23_level_0''' : str(feat_tech_proficiency),

                    '''Unnamed: 24_level_0''' : str(feat_tech_proficiency),

                    '''Unnamed: 25_level_0''' : str(feat_tech_proficiency),

                    '''Unnamed: 26_level_0''' : str(feat_tech_proficiency),

                    '''Unnamed: 27_level_0''' : str(feat_tech_proficiency),

                    '''Unnamed: 28_level_0''' : str(feat_tech_proficiency),

                    '''Unnamed: 29_level_0''' : str(feat_tech_proficiency),

                    '''Unnamed: 30_level_0''' : str(feat_tech_proficiency),

                    '''Unnamed: 31_level_0''' : str(feat_tech_proficiency),

                    '''Unnamed: 32_level_0''' : str(feat_tech_proficiency),

                    '''Unnamed: 33_level_0''' : str(feat_tech_proficiency),

                    '''Unnamed: 34_level_0''' : str(feat_tech_proficiency),

                    '''Unnamed: 35_level_0''' : str(feat_tech_proficiency),

                    '''Unnamed: 36_level_0''' : str(feat_tech_proficiency),

                    '''Which desktop operating system do you use the most?''' : str(feat_os),

                    '''What best describes your career / job satisfaction? ''' : str(feat_career_satisfaction),

                    '''Including bonus, what is your annual compensation in USD?''' : str(feat_annual_salary_plus_bonus),

                    '''Have you visited / Are you aware of Stack Overflow Careers?''' : str(feat_SO_carrer_aware),

                    '''Do you have a Stack Overflow Careers Profile?''' : str(feat_SO_carrer_profile),

                    '''You answered you don't have a Careers profile), can you elaborate why?''' : str(feat_SO_carrer_profile_why_not),

                    '''Unnamed: 43_level_0''' : str(feat_SO_carrer_profile_why_not_other),

                    '''Which technology products do you own? (You can choose more than one)''' : str(feat_gadget),

                    '''Unnamed: 45_level_0''' : str(feat_gadget),

                    '''Unnamed: 46_level_0''' : str(feat_gadget),

                    '''Unnamed: 47_level_0''' : str(feat_gadget),

                    '''Unnamed: 48_level_0''' : str(feat_gadget),

                    '''Unnamed: 49_level_0''' : str(feat_gadget),

                    '''Unnamed: 50_level_0''' : str(feat_gadget),

                    '''Unnamed: 51_level_0''' : str(feat_gadget),

                    '''Unnamed: 52_level_0''' : str(feat_gadget),

                    '''Unnamed: 53_level_0''' : str(feat_gadget),

                    '''Unnamed: 54_level_0''' : str(feat_gadget),

                    '''Unnamed: 55_level_0''' : str(feat_gadget),

                    '''Unnamed: 56_level_0''' : str(feat_gadget),

                    '''Unnamed: 57_level_0''' : str(feat_gadget),

                    '''Unnamed: 58_level_0''' : str(feat_gadget),

                    '''Unnamed: 59_level_0''' : str(feat_gadget),

                    '''Unnamed: 60_level_0''' : str(feat_gadget),

                    '''Unnamed: 61_level_0''' : str(feat_gadget),

                    '''Unnamed: 62_level_0''' : str(feat_gadget),

                    '''Unnamed: 63_level_0''' : str(feat_gadget),

                    '''In the last 12 months, how much money have you spent on personal technology-related purchases? ''' : str(feat_money_tech),

                    '''Please rate the advertising you've seen on Stack Overflow''' : str(feat_ad_rate),

                    '''Unnamed: 66_level_0''' : str(feat_ad_rate),

                    '''Unnamed: 67_level_0''' : str(feat_ad_rate),

                    '''Unnamed: 68_level_0''' : str(feat_ad_rate),

                    '''Unnamed: 69_level_0''' : str(feat_ad_rate),

                    '''Unnamed: 70_level_0''' : str(feat_ad_rate),

                    '''What advertisers do you remember seeing on Stack Overflow?''' : str(feat_ads_seen),

                    '''What is your current Stack Overflow reputation?''' : str(feat_SO_reputation),

                    '''Which of our sites do you frequent most?''' : str(feat_SO_freq_site),

                    '''Unnamed: 74_level_0''' : str(feat_SO_freq_site_other)}
mdf.rename_columns(2012, rename_2012_dict)
mdf.value_counts(2012, feat_company_size)
mdf.value_counts(2013, '''How many people work for your company?''')
feat_company_size_dev = feat_company_size + ' Devs'

feat_team_size = FeatureName('Team Size', 'numerical')

feat_team_interation = FeatureName('Team Interation', 'categorical')

feat_company_mobile_platform = FeatureName('Company Mobile Platform', 'categorical')

feat_company_software_business = FeatureName('Company Software Business', 'categorical')

feat_week_time_spent = FeatureName('Time Spent in Week', 'numerical')

feat_tech_excitement = FeatureName('Exciting Tech', 'categorical')

feat_required_company_qualities = FeatureName('Company Qualities Required', 'categorical')

feat_job_changed_12mos = FeatureName('Changed job 12mos', 'categorical')

feat_SO_usage = FeatureName('StackO Use', 'categorical')



rename_2013_dict = {'''What Country or Region do you live in?''' : str(feat_country),

                    '''Which US State or Territory do you live in?''' : str(feat_us_region),

                    '''How old are you?''' : str(feat_age),

                    '''How many years of IT/Programming experience do you have?''' : str(feat_years_programming),

                    '''How would you best describe the industry you currently work in?''' : str(feat_industry),

                    '''How many people work for your company?''' : str(feat_company_size),

                    '''Which of the following best describes your occupation?''' : str(feat_occupation),

                    '''Including yourself, how many developers are employed at your company?''' : str(feat_company_size_dev),

                    '''How large is the team that you work on?''' : str(feat_team_size),

                    '''What other departments / roles do you interact with regularly?''' : str(feat_team_interation),

                    '''Unnamed: 10_level_0''' : str(feat_team_interation),

                    '''Unnamed: 11_level_0''' : str(feat_team_interation),

                    '''Unnamed: 12_level_0''' : str(feat_team_interation),

                    '''Unnamed: 13_level_0''' : str(feat_team_interation),

                    '''Unnamed: 14_level_0''' : str(feat_team_interation),

                    '''Unnamed: 15_level_0''' : str(feat_team_interation),

                    '''Unnamed: 16_level_0''' : str(feat_team_interation),

                    '''Unnamed: 17_level_0''' : str(feat_team_interation),

                    '''Unnamed: 18_level_0''' : str(feat_team_interation),

                    '''If your company has a native mobile app, what platforms do you support?''' : str(feat_company_mobile_platform),

                    '''Unnamed: 20_level_0''' : str(feat_company_mobile_platform),

                    '''Unnamed: 21_level_0''' : str(feat_company_mobile_platform),

                    '''Unnamed: 22_level_0''' : str(feat_company_mobile_platform),

                    '''Unnamed: 23_level_0''' : str(feat_company_mobile_platform),

                    '''Unnamed: 24_level_0''' : str(feat_company_mobile_platform),

                    '''Unnamed: 25_level_0''' : str(feat_company_mobile_platform),

                    '''If you make a software product, how does your company make money? (You can choose more than one)''' : str(feat_company_software_business),

                    '''Unnamed: 27_level_0''' : str(feat_company_software_business),

                    '''Unnamed: 28_level_0''' : str(feat_company_software_business),

                    '''Unnamed: 29_level_0''' : str(feat_company_software_business),

                    '''Unnamed: 30_level_0''' : str(feat_company_software_business),

                    '''Unnamed: 31_level_0''' : str(feat_company_software_business),

                    '''Unnamed: 32_level_0''' : str(feat_company_software_business),

                    '''Unnamed: 33_level_0''' : str(feat_company_software_business),

                    '''In an average week, how do you spend your time?''' : str(feat_week_time_spent),

                    '''Unnamed: 35_level_0''' : str(feat_week_time_spent),

                    '''Unnamed: 36_level_0''' : str(feat_week_time_spent),

                    '''Unnamed: 37_level_0''' : str(feat_week_time_spent),

                    '''Unnamed: 38_level_0''' : str(feat_week_time_spent),

                    '''Unnamed: 39_level_0''' : str(feat_week_time_spent),

                    '''Unnamed: 40_level_0''' : str(feat_week_time_spent),

                    '''Unnamed: 41_level_0''' : str(feat_week_time_spent),

                    '''What is your involvement in purchasing products or services for the company you work for? (You can choose more than one)''' : str(feat_purch_role),

                    '''Unnamed: 43_level_0''' : str(feat_purch_role),

                    '''Unnamed: 44_level_0''' : str(feat_purch_role),

                    '''Unnamed: 45_level_0''' : str(feat_purch_role),

                    '''Unnamed: 46_level_0''' : str(feat_purch_role),

                    '''Unnamed: 47_level_0''' : str(feat_purch_role),

                    '''Unnamed: 48_level_0''' : str(feat_purch_role),

                    '''What types of purchases are you involved in?''' : str(feat_purch_target),

                    '''Unnamed: 50_level_0''' : str(feat_purch_target),

                    '''Unnamed: 51_level_0''' : str(feat_purch_target),

                    '''Unnamed: 52_level_0''' : str(feat_purch_target),

                    '''Unnamed: 53_level_0''' : str(feat_purch_target),

                    '''Unnamed: 54_level_0''' : str(feat_purch_target),

                    '''What is your budget for outside expenditures (hardware), software), consulting), etc) for 2013?''' : str(feat_budget),

                    '''Which of the following languages or technologies have you used significantly in the past year?''' : str(feat_tech_proficiency),

                    '''Unnamed: 57_level_0''' : str(feat_tech_proficiency),

                    '''Unnamed: 58_level_0''' : str(feat_tech_proficiency),

                    '''Unnamed: 59_level_0''' : str(feat_tech_proficiency),

                    '''Unnamed: 60_level_0''' : str(feat_tech_proficiency),

                    '''Unnamed: 61_level_0''' : str(feat_tech_proficiency),

                    '''Unnamed: 62_level_0''' : str(feat_tech_proficiency),

                    '''Unnamed: 63_level_0''' : str(feat_tech_proficiency),

                    '''Unnamed: 64_level_0''' : str(feat_tech_proficiency),

                    '''Unnamed: 65_level_0''' : str(feat_tech_proficiency),

                    '''Unnamed: 66_level_0''' : str(feat_tech_proficiency),

                    '''Unnamed: 67_level_0''' : str(feat_tech_proficiency),

                    '''Unnamed: 68_level_0''' : str(feat_tech_proficiency),

                    '''Unnamed: 69_level_0''' : str(feat_tech_proficiency),

                    '''Which technologies are you excited about?''' : str(feat_tech_excitement),

                    '''Unnamed: 71_level_0''' : str(feat_tech_excitement),

                    '''Unnamed: 72_level_0''' : str(feat_tech_excitement),

                    '''Unnamed: 73_level_0''' : str(feat_tech_excitement),

                    '''Unnamed: 74_level_0''' : str(feat_tech_excitement),

                    '''Unnamed: 75_level_0''' : str(feat_tech_excitement),

                    '''Unnamed: 76_level_0''' : str(feat_tech_excitement),

                    '''Unnamed: 77_level_0''' : str(feat_tech_excitement),

                    '''Unnamed: 78_level_0''' : str(feat_tech_excitement),

                    '''Unnamed: 79_level_0''' : str(feat_tech_excitement),

                    '''Unnamed: 80_level_0''' : str(feat_tech_excitement),

                    '''Which desktop operating system do you use the most?''' : str(feat_os),

                    '''Please rate how important each of the following characteristics of a company/job offer are to you.    Please select a MAXIMUM of 3 items as "Non-Negotiables" to help us identify the most important items), those where you would never consider a company if they didn't meet them.''' : str(feat_required_company_qualities),

                    '''Unnamed: 83_level_0''' : str(feat_required_company_qualities),

                    '''Unnamed: 84_level_0''' : str(feat_required_company_qualities),

                    '''Unnamed: 85_level_0''' : str(feat_required_company_qualities),

                    '''Unnamed: 86_level_0''' : str(feat_required_company_qualities),

                    '''Unnamed: 87_level_0''' : str(feat_required_company_qualities),

                    '''Unnamed: 88_level_0''' : str(feat_required_company_qualities),

                    '''Unnamed: 89_level_0''' : str(feat_required_company_qualities),

                    '''Unnamed: 90_level_0''' : str(feat_required_company_qualities),

                    '''Unnamed: 91_level_0''' : str(feat_required_company_qualities),

                    '''Unnamed: 92_level_0''' : str(feat_required_company_qualities),

                    '''Unnamed: 93_level_0''' : str(feat_required_company_qualities),

                    '''Unnamed: 94_level_0''' : str(feat_required_company_qualities),

                    '''Unnamed: 95_level_0''' : str(feat_required_company_qualities),

                    '''Unnamed: 96_level_0''' : str(feat_required_company_qualities),

                    '''Unnamed: 97_level_0''' : str(feat_required_company_qualities),

                    '''Have you changed jobs in the last 12 months?''' : str(feat_job_changed_12mos),

                    '''What best describes your career / job satisfaction?''' : str(feat_career_satisfaction),

                    '''Including bonus, what is your annual compensation in USD?''' : str(feat_annual_salary_plus_bonus),

                    '''Which technology products do you own? (You can choose more than one)''' : str(feat_gadget),

                    '''Unnamed: 102_level_0''' : str(feat_gadget),

                    '''Unnamed: 103_level_0''' : str(feat_gadget),

                    '''Unnamed: 104_level_0''' : str(feat_gadget),

                    '''Unnamed: 105_level_0''' : str(feat_gadget),

                    '''Unnamed: 106_level_0''' : str(feat_gadget),

                    '''Unnamed: 107_level_0''' : str(feat_gadget),

                    '''Unnamed: 108_level_0''' : str(feat_gadget),

                    '''Unnamed: 109_level_0''' : str(feat_gadget),

                    '''Unnamed: 110_level_0''' : str(feat_gadget),

                    '''Unnamed: 111_level_0''' : str(feat_gadget),

                    '''Unnamed: 112_level_0''' : str(feat_gadget),

                    '''Unnamed: 113_level_0''' : str(feat_gadget),

                    '''Unnamed: 114_level_0''' : str(feat_gadget),

                    '''In the last 12 months, how much money have you spent on personal technology-related purchases?''' : str(feat_money_tech),

                    '''Please rate the advertising you've seen on Stack Overflow''' : str(feat_ad_rate),

                    '''Unnamed: 117_level_0''' : str(feat_ad_rate),

                    '''Unnamed: 118_level_0''' : str(feat_ad_rate),

                    '''Unnamed: 119_level_0''' : str(feat_ad_rate),

                    '''Unnamed: 120_level_0''' : str(feat_ad_rate),

                    '''Unnamed: 121_level_0''' : str(feat_ad_rate),

                    '''What advertisers do you remember seeing on Stack Overflow?''' : str(feat_ads_seen),

                    '''What is your current Stack Overflow reputation?''' : str(feat_SO_reputation),

                    '''How do you use Stack Overflow?''' : str(feat_SO_usage),

                    '''Unnamed: 125_level_0''' : str(feat_SO_usage),

                    '''Unnamed: 126_level_0''' : str(feat_SO_usage),

                    '''Unnamed: 127_level_0''' : str(feat_SO_usage)}
mdf.rename_columns(2013, rename_2013_dict)
mdf.dfs[2014].describe()
drop_columns_2014 = ['Unnamed: 1_level_0']
drop_columns_2014.append('''Were you aware of the Apptivate contest?''')

drop_columns_2014.append('''Did you participate in the Apptivate contest?''')



mdf.drop_column(2014, drop_columns_2014, level = 0)
mdf.print_empty_rename_map(2014)
feat_gender = FeatureName('Gender', 'categorical')

feat_remote_work = FeatureName('Remote Work', 'numerical')

feat_remote_work_enjoy = FeatureName('Enjoy Remote Work', 'categorical')

feat_remote_work_location = FeatureName('Remote Work Location', 'categorical')

feat_job_discovery = FeatureName('Job Discovery', 'categorical')

feat_job_discovery_other = FeatureName('Job Discovery Other', 'categorical')

feat_job_searching = FeatureName('Searching for Job', 'categorical')

feat_recruters_contact_freq = FeatureName('Recruters Contact Freq.', 'numerical')

feat_job_contact_channel = FeatureName('Job Contact Channel', 'categorical')

feat_job_msg_appealing = FeatureName('Prefered Info in Job Mail', 'categorical')

feat_job_board = FeatureName('Job Board', 'categorical')

feat_find_not_ask = FeatureName('''Find solution without asking''', 'categorical')



rename_2014_dict = {'''What Country do you live in?''' : str(feat_country),

                    '''Which US State or Territory do you live in?''' : str(feat_us_region),

                    '''How old are you?''' : str(feat_age),

                    '''What is your gender?''' : str(feat_gender),

                    '''How many years of IT/Programming experience do you have?''' : str(feat_years_programming),

                    '''Which of the following best describes your occupation?''' : str(feat_occupation),

                    '''Including bonus, what is your annual compensation in USD?''' : str(feat_annual_salary_plus_bonus),

                    '''How would you best describe the industry you currently work in?''' : str(feat_industry),

                    '''How many developers are employed at your company?''' : str(feat_company_size_dev),

                    '''Do you work remotely?''' : str(feat_remote_work),

                    '''Do you enjoy working remotely?''' : str(feat_remote_work_enjoy),

                    '''Where do you work remotely most of the time?''' : str(feat_remote_work_location),

                    '''If your company has a native mobile app, what platforms do you support?''' : str(feat_remote_work_location),

                    '''Unnamed: 14_level_0''' : str(feat_remote_work_location),

                    '''Unnamed: 15_level_0''' : str(feat_remote_work_location),

                    '''Unnamed: 16_level_0''' : str(feat_remote_work_location),

                    '''Unnamed: 17_level_0''' : str(feat_remote_work_location),

                    '''Unnamed: 18_level_0''' : str(feat_remote_work_location),

                    '''Unnamed: 19_level_0''' : str(feat_remote_work_location),

                    '''In an average week, how do you spend your time at work?''' : str(feat_week_time_spent),

                    '''Unnamed: 21_level_0''' : str(feat_week_time_spent),

                    '''Unnamed: 22_level_0''' : str(feat_week_time_spent),

                    '''Unnamed: 23_level_0''' : str(feat_week_time_spent),

                    '''Unnamed: 24_level_0''' : str(feat_week_time_spent),

                    '''Unnamed: 25_level_0''' : str(feat_week_time_spent),

                    '''Unnamed: 26_level_0''' : str(feat_week_time_spent),

                    '''Unnamed: 27_level_0''' : str(feat_week_time_spent),

                    '''Unnamed: 28_level_0''' : str(feat_week_time_spent),

                    '''What is your involvement in purchasing products or services for the company you work for? (You can choose more than one)''' : str(feat_purch_role),

                    '''Unnamed: 30_level_0''' : str(feat_purch_role),

                    '''Unnamed: 31_level_0''' : str(feat_purch_role),

                    '''Unnamed: 32_level_0''' : str(feat_purch_role),

                    '''Unnamed: 33_level_0''' : str(feat_purch_role),

                    '''What types of purchases are you involved in?''' : str(feat_purch_target),

                    '''Unnamed: 35_level_0''' : str(feat_purch_target),

                    '''Unnamed: 36_level_0''' : str(feat_purch_target),

                    '''Unnamed: 37_level_0''' : str(feat_purch_target),

                    '''Unnamed: 38_level_0''' : str(feat_purch_target),

                    '''Unnamed: 39_level_0''' : str(feat_purch_target),

                    '''Unnamed: 40_level_0''' : str(feat_purch_target),

                    '''What is your budget for outside expenditures (hardware, software, consulting, etc) for 2014?''' : str(feat_budget),

                    '''Which of the following languages or technologies have you used significantly in the past year?''' : str(feat_tech_proficiency),

                    '''Unnamed: 43_level_0''' : str(feat_tech_proficiency),

                    '''Unnamed: 44_level_0''' : str(feat_tech_proficiency),

                    '''Unnamed: 45_level_0''' : str(feat_tech_proficiency),

                    '''Unnamed: 46_level_0''' : str(feat_tech_proficiency),

                    '''Unnamed: 47_level_0''' : str(feat_tech_proficiency),

                    '''Unnamed: 48_level_0''' : str(feat_tech_proficiency),

                    '''Unnamed: 49_level_0''' : str(feat_tech_proficiency),

                    '''Unnamed: 50_level_0''' : str(feat_tech_proficiency),

                    '''Unnamed: 51_level_0''' : str(feat_tech_proficiency),

                    '''Unnamed: 52_level_0''' : str(feat_tech_proficiency),

                    '''Unnamed: 53_level_0''' : str(feat_tech_proficiency),

                    '''Which technologies are you excited about?''' : str(feat_tech_excitement),

                    '''Unnamed: 55_level_0''' : str(feat_tech_excitement),

                    '''Unnamed: 56_level_0''' : str(feat_tech_excitement),

                    '''Unnamed: 57_level_0''' : str(feat_tech_excitement),

                    '''Unnamed: 58_level_0''' : str(feat_tech_excitement),

                    '''Unnamed: 59_level_0''' : str(feat_tech_excitement),

                    '''Unnamed: 60_level_0''' : str(feat_tech_excitement),

                    '''Unnamed: 61_level_0''' : str(feat_tech_excitement),

                    '''Unnamed: 62_level_0''' : str(feat_tech_excitement),

                    '''Unnamed: 63_level_0''' : str(feat_tech_excitement),

                    '''Unnamed: 64_level_0''' : str(feat_tech_excitement),

                    '''Unnamed: 65_level_0''' : str(feat_tech_excitement),

                    '''Unnamed: 66_level_0''' : str(feat_tech_excitement),

                    '''Which desktop operating system do you use the most?''' : str(feat_os),

                    '''Which technology products do you own? (You can choose more than one)''' : str(feat_gadget),

                    '''Unnamed: 69_level_0''' : str(feat_gadget),

                    '''Unnamed: 70_level_0''' : str(feat_gadget),

                    '''Unnamed: 71_level_0''' : str(feat_gadget),

                    '''Unnamed: 72_level_0''' : str(feat_gadget),

                    '''Unnamed: 73_level_0''' : str(feat_gadget),

                    '''Unnamed: 74_level_0''' : str(feat_gadget),

                    '''Unnamed: 75_level_0''' : str(feat_gadget),

                    '''Unnamed: 76_level_0''' : str(feat_gadget),

                    '''Unnamed: 77_level_0''' : str(feat_gadget),

                    '''Unnamed: 78_level_0''' : str(feat_gadget),

                    '''Unnamed: 79_level_0''' : str(feat_gadget),

                    '''Unnamed: 80_level_0''' : str(feat_gadget),

                    '''Unnamed: 81_level_0''' : str(feat_gadget),

                    '''Have you changed jobs in the last 12 months?''' : str(feat_job_changed_12mos),

                    '''How did you find out about your current job?''' : str(feat_job_discovery),

                    '''Unnamed: 84_level_0''' : str(feat_job_discovery_other),

                    '''Are you currently looking for a job or open to new opportunities?''' : str(feat_job_searching),

                    '''How often are you contacted by recruiters?''' : str(feat_recruters_contact_freq),

                    '''How do you prefer to be contacted about job opportunities?''' : str(feat_job_contact_channel),

                    '''Unnamed: 88_level_0''' : str(feat_job_contact_channel),

                    '''Unnamed: 89_level_0''' : str(feat_job_contact_channel),

                    '''Unnamed: 90_level_0''' : str(feat_job_contact_channel),

                    '''Unnamed: 91_level_0''' : str(feat_job_contact_channel),

                    '''In receiving an email about a job opportunity), what attributes of the message would make you more likely to respond?''' : str(feat_job_msg_appealing),

                    '''Unnamed: 93_level_0''' : str(feat_job_msg_appealing),

                    '''Unnamed: 94_level_0''' : str(feat_job_msg_appealing),

                    '''Unnamed: 95_level_0''' : str(feat_job_msg_appealing),

                    '''Unnamed: 96_level_0''' : str(feat_job_msg_appealing),

                    '''Unnamed: 97_level_0''' : str(feat_job_msg_appealing),

                    '''Unnamed: 98_level_0''' : str(feat_job_msg_appealing),

                    '''How often do you visit job boards?''' : str(feat_job_board),

                    '''Have you visited / Are you aware of Stack Overflow Careers 2.0?''' : str(feat_SO_carrer_aware),

                    '''Do you have a Stack Overflow Careers 2.0 Profile?''' : str(feat_SO_carrer_profile),

                    '''Please rate the advertising you've seen on Stack Overflow''' : str(feat_ad_rate),

                    '''Unnamed: 103_level_0''' : str(feat_ad_rate),

                    '''Unnamed: 104_level_0''' : str(feat_ad_rate),

                    '''Unnamed: 105_level_0''' : str(feat_ad_rate),

                    '''Unnamed: 106_level_0''' : str(feat_ad_rate),

                    '''Unnamed: 107_level_0''' : str(feat_ad_rate),

                    '''Unnamed: 108_level_0''' : str(feat_ad_rate),

                    '''What advertisers do you remember seeing on Stack Overflow?''' : str(feat_ads_seen),

                    '''What is your current Stack Overflow reputation?''' : str(feat_SO_reputation),

                    '''How do you use Stack Overflow?''' : str(feat_SO_usage),

                    '''Unnamed: 114_level_0''' : str(feat_SO_usage),

                    '''Unnamed: 115_level_0''' : str(feat_SO_usage),

                    '''Unnamed: 116_level_0''' : str(feat_SO_usage),

                    '''Unnamed: 117_level_0''' : str(feat_SO_usage),

                    '''Unnamed: 118_level_0''' : str(feat_SO_usage),

                    '''How often do you find solutions to your programming problems on Stack Overflow without asking a new question?''' : str(feat_find_not_ask)}
mdf.rename_columns(2014, rename_2014_dict)
mdf.print_df_column_names(2015)
new_names_2015_lv1 = mdf.get_column_names(2015, level=1)

new_names_2015_lv0 = ['lv0_{}'.format(name) for name in new_names_2015_lv1]



mdf.set_column_multiIndex(2015, [new_names_2015_lv0, new_names_2015_lv1])
mdf.value_counts(2015, ['lv0_Compensation: midpoint','lv0_Compensation'])
mdf.drop_column(2015, '''lv0_Compensation: midpoint''', level = 0)
mdf.print_df_column_names(2015)
mdf.print_empty_rename_map(2015)
feat_tab_space = FeatureName( 'Tabs or Spaces', 'categorical')

feat_os_other = feat_os + ' other'

feat_employment_status = FeatureName( 'Employment Status', 'categorical')

feat_training_edu = FeatureName( 'Training & Education', 'categorical')

feat_job_satisfaction = FeatureName( 'Job Satisfaction', 'numerical')

feat_job_wanted_benefits = FeatureName( 'Job Wanted Benefits', 'categorical')

feat_remote_work_relevance = FeatureName( 'Remote Work Relevance', 'numerical')

feat_job_search_annoyance = FeatureName( 'Annoyance in Job Searching', 'categorical')

feat_recruiter_contact_perception = FeatureName( 'Perception of recruiter contact', 'categorical')

feat_job_opportunity_urgent_info = FeatureName( 'Urgent Info About Job Opportunity', 'categorical')

feat_job_opportunity_contact_person = FeatureName( 'Prefered to talk about job opportunity', 'categorical')

feat_how_to_improve_interview = FeatureName( 'How to improve interviews', 'categorical')

feat_SO_career_why = FeatureName( 'Why try Stack Overflow Careers', 'categorical')

feat_caffeinated_beverages_day = FeatureName( 'caffeinated beverages per day', 'numerical')

feat_program_hobby_hours_week = FeatureName( 'program as hobby hours/week', 'numerical')



feat_text_editor = FeatureName( 'Preferred text editor', 'categorical')

feat_text_editor_other = feat_text_editor + ' other'

feat_IDE_theme = FeatureName( 'Prefered IDE theme', 'categorical')

feat_source_control_used = FeatureName( 'Source Control Used', 'categorical')

feat_source_control_used_other = feat_source_control_used + ' other'

feat_source_control_prefered = FeatureName( 'Prefered Source Control', 'categorical')

feat_source_control_prefered_other = feat_source_control_prefered + ' other'

feat_SO_why_use = FeatureName( 'Why use StackO', 'categorical')

feat_SO_freq_helpful = FeatureName( 'How often are the answers helpful', 'numerical')

feat_SO_why_helpful = FeatureName( 'Why answers are helpful', 'categorical')



rename_2015_dict = {'''lv0_Country''' : str(feat_country),

                    '''lv0_Age''' : str(feat_age),

                    '''lv0_Gender''' : str(feat_gender),

                    '''lv0_Tabs or Spaces''' : str(feat_tab_space),

                    '''lv0_Years IT / Programming Experience''' : str(feat_years_programming),

                    '''lv0_Occupation''' : str(feat_occupation),

                    '''lv0_Desktop Operating System''' : str(feat_os),

                    '''lv0_Desktop Operating System: write-in''' : str(feat_os_other),

                    '''lv0_Current Lang & Tech: Android''' : str(feat_tech_proficiency),

                    '''lv0_Current Lang & Tech: Arduino''' : str(feat_tech_proficiency),

                    '''lv0_Current Lang & Tech: AngularJS''' : str(feat_tech_proficiency),

                    '''lv0_Current Lang & Tech: C''' : str(feat_tech_proficiency),

                    '''lv0_Current Lang & Tech: C++''' : str(feat_tech_proficiency),

                    '''lv0_Current Lang & Tech: C++11''' : str(feat_tech_proficiency),

                    '''lv0_Current Lang & Tech: C#''' : str(feat_tech_proficiency),

                    '''lv0_Current Lang & Tech: Cassandra''' : str(feat_tech_proficiency),

                    '''lv0_Current Lang & Tech: CoffeeScript''' : str(feat_tech_proficiency),

                    '''lv0_Current Lang & Tech: Cordova''' : str(feat_tech_proficiency),

                    '''lv0_Current Lang & Tech: Clojure''' : str(feat_tech_proficiency),

                    '''lv0_Current Lang & Tech: Cloud''' : str(feat_tech_proficiency),

                    '''lv0_Current Lang & Tech: Dart''' : str(feat_tech_proficiency),

                    '''lv0_Current Lang & Tech: F#''' : str(feat_tech_proficiency),

                    '''lv0_Current Lang & Tech: Go''' : str(feat_tech_proficiency),

                    '''lv0_Current Lang & Tech: Hadoop''' : str(feat_tech_proficiency),

                    '''lv0_Current Lang & Tech: Haskell''' : str(feat_tech_proficiency),

                    '''lv0_Current Lang & Tech: iOS''' : str(feat_tech_proficiency),

                    '''lv0_Current Lang & Tech: Java''' : str(feat_tech_proficiency),

                    '''lv0_Current Lang & Tech: JavaScript''' : str(feat_tech_proficiency),

                    '''lv0_Current Lang & Tech: LAMP''' : str(feat_tech_proficiency),

                    '''lv0_Current Lang & Tech: Matlab''' : str(feat_tech_proficiency),

                    '''lv0_Current Lang & Tech: MongoDB''' : str(feat_tech_proficiency),

                    '''lv0_Current Lang & Tech: Node.js''' : str(feat_tech_proficiency),

                    '''lv0_Current Lang & Tech: Objective-C''' : str(feat_tech_proficiency),

                    '''lv0_Current Lang & Tech: Perl''' : str(feat_tech_proficiency),

                    '''lv0_Current Lang & Tech: PHP''' : str(feat_tech_proficiency),

                    '''lv0_Current Lang & Tech: Python''' : str(feat_tech_proficiency),

                    '''lv0_Current Lang & Tech: R''' : str(feat_tech_proficiency),

                    '''lv0_Current Lang & Tech: Redis''' : str(feat_tech_proficiency),

                    '''lv0_Current Lang & Tech: Ruby''' : str(feat_tech_proficiency),

                    '''lv0_Current Lang & Tech: Rust''' : str(feat_tech_proficiency),

                    '''lv0_Current Lang & Tech: Salesforce''' : str(feat_tech_proficiency),

                    '''lv0_Current Lang & Tech: Scala''' : str(feat_tech_proficiency),

                    '''lv0_Current Lang & Tech: Sharepoint''' : str(feat_tech_proficiency),

                    '''lv0_Current Lang & Tech: Spark''' : str(feat_tech_proficiency),

                    '''lv0_Current Lang & Tech: SQL''' : str(feat_tech_proficiency),

                    '''lv0_Current Lang & Tech: SQL Server''' : str(feat_tech_proficiency),

                    '''lv0_Current Lang & Tech: Swift''' : str(feat_tech_proficiency),

                    '''lv0_Current Lang & Tech: Visual Basic''' : str(feat_tech_proficiency),

                    '''lv0_Current Lang & Tech: Windows Phone''' : str(feat_tech_proficiency),

                    '''lv0_Current Lang & Tech: Wordpress''' : str(feat_tech_proficiency),

                    '''lv0_Current Lang & Tech: Write-In''' : str(feat_tech_proficiency),

                    '''lv0_Future Lang & Tech: Android''' : str(feat_tech_excitement),

                    '''lv0_Future Lang & Tech: Arduino''' : str(feat_tech_excitement),

                    '''lv0_Future Lang & Tech: AngularJS''' : str(feat_tech_excitement),

                    '''lv0_Future Lang & Tech: C''' : str(feat_tech_excitement),

                    '''lv0_Future Lang & Tech: C++''' : str(feat_tech_excitement),

                    '''lv0_Future Lang & Tech: C++11''' : str(feat_tech_excitement),

                    '''lv0_Future Lang & Tech: C#''' : str(feat_tech_excitement),

                    '''lv0_Future Lang & Tech: Cassandra''' : str(feat_tech_excitement),

                    '''lv0_Future Lang & Tech: CoffeeScript''' : str(feat_tech_excitement),

                    '''lv0_Future Lang & Tech: Cordova''' : str(feat_tech_excitement),

                    '''lv0_Future Lang & Tech: Clojure''' : str(feat_tech_excitement),

                    '''lv0_Future Lang & Tech: Cloud''' : str(feat_tech_excitement),

                    '''lv0_Future Lang & Tech: Dart''' : str(feat_tech_excitement),

                    '''lv0_Future Lang & Tech: F#''' : str(feat_tech_excitement),

                    '''lv0_Future Lang & Tech: Go''' : str(feat_tech_excitement),

                    '''lv0_Future Lang & Tech: Hadoop''' : str(feat_tech_excitement),

                    '''lv0_Future Lang & Tech: Haskell''' : str(feat_tech_excitement),

                    '''lv0_Future Lang & Tech: iOS''' : str(feat_tech_excitement),

                    '''lv0_Future Lang & Tech: Java''' : str(feat_tech_excitement),

                    '''lv0_Future Lang & Tech: JavaScript''' : str(feat_tech_excitement),

                    '''lv0_Future Lang & Tech: LAMP''' : str(feat_tech_excitement),

                    '''lv0_Future Lang & Tech: Matlab''' : str(feat_tech_excitement),

                    '''lv0_Future Lang & Tech: MongoDB''' : str(feat_tech_excitement),

                    '''lv0_Future Lang & Tech: Node.js''' : str(feat_tech_excitement),

                    '''lv0_Future Lang & Tech: Objective-C''' : str(feat_tech_excitement),

                    '''lv0_Future Lang & Tech: Perl''' : str(feat_tech_excitement),

                    '''lv0_Future Lang & Tech: PHP''' : str(feat_tech_excitement),

                    '''lv0_Future Lang & Tech: Python''' : str(feat_tech_excitement),

                    '''lv0_Future Lang & Tech: R''' : str(feat_tech_excitement),

                    '''lv0_Future Lang & Tech: Redis''' : str(feat_tech_excitement),

                    '''lv0_Future Lang & Tech: Ruby''' : str(feat_tech_excitement),

                    '''lv0_Future Lang & Tech: Rust''' : str(feat_tech_excitement),

                    '''lv0_Future Lang & Tech: Salesforce''' : str(feat_tech_excitement),

                    '''lv0_Future Lang & Tech: Scala''' : str(feat_tech_excitement),

                    '''lv0_Future Lang & Tech: Sharepoint''' : str(feat_tech_excitement),

                    '''lv0_Future Lang & Tech: Spark''' : str(feat_tech_excitement),

                    '''lv0_Future Lang & Tech: SQL''' : str(feat_tech_excitement),

                    '''lv0_Future Lang & Tech: SQL Server''' : str(feat_tech_excitement),

                    '''lv0_Future Lang & Tech: Swift''' : str(feat_tech_excitement),

                    '''lv0_Future Lang & Tech: Visual Basic''' : str(feat_tech_excitement),

                    '''lv0_Future Lang & Tech: Windows Phone''' : str(feat_tech_excitement),

                    '''lv0_Future Lang & Tech: Wordpress''' : str(feat_tech_excitement),

                    '''lv0_Future Lang & Tech: Write-In''' : str(feat_tech_excitement),

                    '''lv0_Training & Education: No formal training''' : str(feat_training_edu),

                    '''lv0_Training & Education: On the job''' : str(feat_training_edu),

                    '''lv0_Training & Education: Boot camp or night school''' : str(feat_training_edu),

                    '''lv0_Training & Education: Online Class''' : str(feat_training_edu),

                    '''lv0_Training & Education: Mentorship''' : str(feat_training_edu),

                    '''lv0_Training & Education: Industry certification''' : str(feat_training_edu),

                    '''lv0_Training & Education: Some college, but no CS degree''' : str(feat_training_edu),

                    '''lv0_Training & Education: BS in CS''' : str(feat_training_edu),

                    '''lv0_Training & Education: Masters in CS''' : str(feat_training_edu),

                    '''lv0_Training & Education: PhD in CS''' : str(feat_training_edu),

                    '''lv0_Training & Education: Other''' : str(feat_training_edu),

                    '''lv0_Compensation''' : str(feat_annual_salary_plus_bonus),

                    '''lv0_Employment Status''' : str(feat_employment_status),

                    '''lv0_Industry''' : str(feat_industry),

                    '''lv0_Job Satisfaction''' : str(feat_job_satisfaction),

                    '''lv0_Purchasing Power''' : str(feat_purch_role),

                    '''lv0_Remote Status''' : str(feat_remote_work),

                    '''lv0_Changed Jobs in last 12 Months''' : str(feat_job_changed_12mos),

                    '''lv0_Open to new job opportunities''' : str(feat_job_searching),

                    '''lv0_Most important aspect of new job opportunity: Salary''' : str(feat_job_wanted_benefits),

                    '''lv0_Most important aspect of new job opportunity: Equity''' : str(feat_job_wanted_benefits),

                    '''lv0_Most important aspect of new job opportunity: Important decisions''' : str(feat_job_wanted_benefits),

                    '''lv0_Most important aspect of new job opportunity: Health insurance''' : str(feat_job_wanted_benefits),

                    '''lv0_Most important aspect of new job opportunity: Industry''' : str(feat_job_wanted_benefits),

                    '''lv0_Most important aspect of new job opportunity: Tech stack''' : str(feat_job_wanted_benefits),

                    '''lv0_Most important aspect of new job opportunity: Company size''' : str(feat_job_wanted_benefits),

                    '''lv0_Most important aspect of new job opportunity: Company stage''' : str(feat_job_wanted_benefits),

                    '''lv0_Most important aspect of new job opportunity: Work - Life balance''' : str(feat_job_wanted_benefits),

                    '''lv0_Most important aspect of new job opportunity: Advancement''' : str(feat_job_wanted_benefits),

                    '''lv0_Most important aspect of new job opportunity: Job title''' : str(feat_job_wanted_benefits),

                    '''lv0_Most important aspect of new job opportunity: Office location''' : str(feat_job_wanted_benefits),

                    '''lv0_Most important aspect of new job opportunity: Quality of colleagues''' : str(feat_job_wanted_benefits),

                    '''lv0_Most important aspect of new job opportunity: Company culture''' : str(feat_job_wanted_benefits),

                    '''lv0_Most important aspect of new job opportunity: Company reputation''' : str(feat_job_wanted_benefits),

                    '''lv0_Most important aspect of new job opportunity: Building something that matters''' : str(feat_job_wanted_benefits),

                    '''lv0_Most important aspect of new job opportunity: Remote working''' : str(feat_job_wanted_benefits),

                    '''lv0_Most important aspect of new job opportunity: Flexible work options''' : str(feat_job_wanted_benefits),

                    '''lv0_How important is remote when evaluating new job opportunity?''' : str(feat_remote_work_relevance),

                    '''lv0_Most annoying about job search: Finding time''' : str(feat_job_search_annoyance),

                    '''lv0_Most annoying about job search: Finding job I'm qualified for''' : str(feat_job_search_annoyance),

                    '''lv0_Most annoying about job search: Finding interesting job''' : str(feat_job_search_annoyance),

                    '''lv0_Most annoying about job search: Interesting companies rarely respond''' : str(feat_job_search_annoyance),

                    '''lv0_Most annoying about job search: Writing and updating CV''' : str(feat_job_search_annoyance),

                    '''lv0_Most annoying about job search: Taking time off work to interview''' : str(feat_job_search_annoyance),

                    '''lv0_Most annoying about job search: The Interview''' : str(feat_job_search_annoyance),

                    '''lv0_How often contacted by recruiters''' : str(feat_recruters_contact_freq),

                    '''lv0_Perception of recruiter contact''' : str(feat_recruiter_contact_perception),

                    '''lv0_Perception of contact form: Email''' : str(feat_recruiter_contact_perception),

                    '''lv0_Perception of contact form: LinkedIn''' : str(feat_recruiter_contact_perception),

                    '''lv0_Perception of contact form: Xing''' : str(feat_recruiter_contact_perception),

                    '''lv0_Perception of contact form: Phone''' : str(feat_recruiter_contact_perception),

                    '''lv0_Perception of contact form: Stack Overflow Careers''' : str(feat_recruiter_contact_perception),

                    '''lv0_Perception of contact form: Twitter''' : str(feat_recruiter_contact_perception),

                    '''lv0_Perception of contact form: Facebook''' : str(feat_recruiter_contact_perception),

                    '''lv0_Appealing message traits: Message is personalized''' : str(feat_job_msg_appealing),

                    '''lv0_Appealing message traits: Code or projects mentioned''' : str(feat_job_msg_appealing),

                    '''lv0_Appealing message traits: Stack Overflow activity mentioned''' : str(feat_job_msg_appealing),

                    '''lv0_Appealing message traits: Team described''' : str(feat_job_msg_appealing),

                    '''lv0_Appealing message traits: Company culture described''' : str(feat_job_msg_appealing),

                    '''lv0_Appealing message traits: Salary information''' : str(feat_job_msg_appealing),

                    '''lv0_Appealing message traits: Benefits & Perks''' : str(feat_job_msg_appealing),

                    '''lv0_Appealing message traits: Stack Overflow Company Page''' : str(feat_job_msg_appealing),

                    '''lv0_Most urgent info about job opportunity: Salary''' : str(feat_job_opportunity_urgent_info),

                    '''lv0_Most urgent info about job opportunity: Benefits''' : str(feat_job_opportunity_urgent_info),

                    '''lv0_Most urgent info about job opportunity: Company name''' : str(feat_job_opportunity_urgent_info),

                    '''lv0_Most urgent info about job opportunity: Tech stack''' : str(feat_job_opportunity_urgent_info),

                    '''lv0_Most urgent info about job opportunity: Office location''' : str(feat_job_opportunity_urgent_info),

                    '''lv0_Most urgent info about job opportunity: Job title''' : str(feat_job_opportunity_urgent_info),

                    '''lv0_Most urgent info about job opportunity: Colleagues''' : str(feat_job_opportunity_urgent_info),

                    '''lv0_Most urgent info about job opportunity: Product details''' : str(feat_job_opportunity_urgent_info),

                    '''lv0_Who do you want to communicate with about a new job opportunity: Headhunter''' : str(feat_job_opportunity_contact_person),

                    '''lv0_Who do you want to communicate with about a new job opportunity: In-house recruiter''' : str(feat_job_opportunity_contact_person),

                    '''lv0_Who do you want to communicate with about a new job opportunity: In-house tech recruiter''' : str(feat_job_opportunity_contact_person),

                    '''lv0_Who do you want to communicate with about a new job opportunity: Manager''' : str(feat_job_opportunity_contact_person),

                    '''lv0_Who do you want to communicate with about a new job opportunity: Developer''' : str(feat_job_opportunity_contact_person),

                    '''lv0_How can companies improve interview process: More live code''' : str(feat_how_to_improve_interview),

                    '''lv0_How can companies improve interview process: Flexible interview schedule''' : str(feat_how_to_improve_interview),

                    '''lv0_How can companies improve interview process: Remote interviews''' : str(feat_how_to_improve_interview),

                    '''lv0_How can companies improve interview process: Introduce me to boss''' : str(feat_how_to_improve_interview),

                    '''lv0_How can companies improve interview process: Introduce me to team''' : str(feat_how_to_improve_interview),

                    '''lv0_How can companies improve interview process: Gimme coffee''' : str(feat_how_to_improve_interview),

                    '''lv0_How can companies improve interview process: Show me workplace''' : str(feat_how_to_improve_interview),

                    '''lv0_How can companies improve interview process: Fewer brainteasers''' : str(feat_how_to_improve_interview),

                    '''lv0_How can companies improve interview process: Better preparation''' : str(feat_how_to_improve_interview),

                    '''lv0_Why try Stack Overflow Careers: No spam''' : str(feat_SO_career_why),

                    '''lv0_Why try Stack Overflow Careers: Jobs site for programmers''' : str(feat_SO_career_why),

                    '''lv0_Why try Stack Overflow Careers: Selection of revelant jobs''' : str(feat_SO_career_why),

                    '''lv0_Why try Stack Overflow Careers: Showcase Stack Overflow activity''' : str(feat_SO_career_why),

                    '''lv0_Why try Stack Overflow Careers: Jobs are on Stack Overflow''' : str(feat_SO_career_why),

                    '''lv0_Why try Stack Overflow Careers: Other''' : str(feat_SO_career_why),

                    '''lv0_How many caffeinated beverages per day?''' : str(feat_caffeinated_beverages_day),

                    '''lv0_How many hours programming as hobby per week?''' : str(feat_program_hobby_hours_week),

                    '''lv0_How frequently land on or read Stack Overflow''' : str(feat_SO_freq_site),

                    '''lv0_Preferred text editor''' : str(feat_text_editor),

                    '''lv0_Preferred text editor: write-in''' : str(feat_text_editor_other),

                    '''lv0_Prefered IDE theme''' : str(feat_IDE_theme),

                    '''lv0_Source control used: Git''' : str(feat_source_control_used),

                    '''lv0_Source control used: Mercurial''' : str(feat_source_control_used),

                    '''lv0_Source control used: SVN''' : str(feat_source_control_used),

                    '''lv0_Source control used: CVS''' : str(feat_source_control_used),

                    '''lv0_Source control used: Perforce''' : str(feat_source_control_used),

                    '''lv0_Source control used: TFS''' : str(feat_source_control_used),

                    '''lv0_Source control used: DCVS''' : str(feat_source_control_used),

                    '''lv0_Source control used: Bitkeeper''' : str(feat_source_control_used),

                    '''lv0_Source control used: Legacy / Custom''' : str(feat_source_control_used),

                    '''lv0_Source control used: I don't use source control''' : str(feat_source_control_used),

                    '''lv0_Source control used: write-in''' : str(feat_source_control_used_other),

                    '''lv0_Prefered Source Control''' : str(feat_source_control_prefered),

                    '''lv0_Prefered Source Control: write-in''' : str(feat_source_control_prefered_other),

                    '''lv0_Why use Stack Overflow: Help for job''' : str(feat_SO_why_use),

                    '''lv0_Why use Stack Overflow: To give help''' : str(feat_SO_why_use),

                    '''lv0_Why use Stack Overflow: Can't do job without it''' : str(feat_SO_why_use),

                    '''lv0_Why use Stack Overflow: Maintain online presence''' : str(feat_SO_why_use),

                    '''lv0_Why use Stack Overflow: Demonstrate expertise''' : str(feat_SO_why_use),

                    '''lv0_Why use Stack Overflow: Communicate with others''' : str(feat_SO_why_use),

                    '''lv0_Why use Stack Overflow: Receive help on personal projects''' : str(feat_SO_why_use),

                    '''lv0_Why use Stack Overflow: Love to learn''' : str(feat_SO_why_use),

                    '''lv0_Why use Stack Overflow: I don't use Stack Overflow''' : str(feat_SO_why_use),

                    '''lv0_How often are Stack Overflow's answers helpful''' : str(feat_SO_freq_helpful),

                    '''lv0_Why answer: Help a programmer in need''' : str(feat_SO_why_helpful),

                    '''lv0_Why answer: Help future programmers''' : str(feat_SO_why_helpful),

                    '''lv0_Why answer: Demonstrate expertise''' : str(feat_SO_why_helpful),

                    '''lv0_Why answer: Self promotion''' : str(feat_SO_why_helpful),

                    '''lv0_Why answer: Sense of responsibility to developers''' : str(feat_SO_why_helpful),

                    '''lv0_Why answer: No idea''' : str(feat_SO_why_helpful),

                    '''lv0_Why answer: I don't answer and I don't want to''' : str(feat_SO_why_helpful),

                    '''lv0_Why answer: I don't answer but I want to''' : str(feat_SO_why_helpful)}
mdf.rename_columns(2015, rename_2015_dict)
mdf.print_df_column_names(2016)
drop_columns_2016 = {'Unnamed: 0'}
drop_columns_2016.add('age_midpoint')

drop_columns_2016.add('experience_midpoint')

drop_columns_2016.add('salary_midpoint')
mdf.drop_column(2016, drop_columns_2016)
if type(mdf.dfs[2016].columns) is pd.Index:

    new_names_2016_lv1 = mdf.get_column_names(2016)

    new_names_2016_lv0 = ["lv0_{}".format(lv1_name) for lv1_name in new_names_2016_lv1]



    mdf.set_column_multiIndex(2016, [new_names_2016_lv0, new_names_2016_lv1])

mdf.dfs[2016]
mdf.dfs[2016]
mdf.value_counts(2016, '''lv0_agree_mars''')
mdf.value_counts(2016, '''lv0_self_identification''')
mdf.print_empty_rename_map(2016)
feat_collector = FeatureName('Collector', 'categorical')

feat_un_region = FeatureName('UN Region', 'categorical')

feat_SO_region = FeatureName('StackO Region', 'categorical')

feat_self_identification = FeatureName('Self Identification', 'categorical')

feat_occupation_group = FeatureName('Occupation Group', 'categorical')

feat_big_mac_index = FeatureName('Big Mac Index', 'numerical')

feat_aliens = FeatureName('Believe in Aliens', 'categorical')

feat_programming_ability = FeatureName('Programming Ability', 'categorical')

feat_team_has_woman = FeatureName('Team Has Woman', 'categorical')

feat_IDE = FeatureName('IDE', 'categorical')

feat_commit_frequency = FeatureName("Commit Frequency", 'numerical')

feat_dog_cat = FeatureName('Dogs or cats', 'categorical')

feat_unit_testing_worth = FeatureName('Unit Testing Worth', 'categorical')

feat_why_learn_new_tech = FeatureName('Why Learn New Tech', 'categorical')

feat_google_interview_likelihood = FeatureName('Likelihood of Interview at Google', 'numerical')

feat_star_wars_vs_star_trek = FeatureName('Star Wars or Star Trek', 'categorical')

feat_enjoy_tech_using = FeatureName('Enjoy Tech Using at Work', 'categorical')

feat_software_must_improve = FeatureName('Software Must Improve', 'categorical')

feat_like_solving_problem = FeatureName('Like Solving Problems', 'categorical')

feat_diversity_important = FeatureName('Diversity is Important at Work', 'categorical')

feat_use_adblock = FeatureName('Use Ad Blocker', 'categorical')

feat_drink_and_code = FeatureName('Drink and Code', 'categorical')

feat_lovely_boss = FeatureName('Lovely Boss', 'categorical')

feat_love_late_night_coding = FeatureName('Love coding at late night', 'categorical')

feat_legacy_code = FeatureName('Work With Legacy Code', 'categorical')

feat_wanto_go_mars = FeatureName('Want to go to Mars', 'categorical')

feat_important_variety = FeatureName('Important Variety Projects', 'categorical')

feat_important_product_decision = FeatureName('Important to decide over product', 'categorical')

feat_important_leave_same_time = FeatureName('Important to End Word Same Time', 'categorical')

feat_important_learn_tech = FeatureName('Important to Learn New Tech', 'categorical')

feat_important_build_something = FeatureName('Important to Build Something', 'categorical')

feat_important_improve_apps = FeatureName('Important to improve existing apps', 'categorical')

feat_important_promotions = FeatureName('Important to Be Promoted', 'categorical')

feat_important_company_mission = FeatureName('''Believe in Company's Mission''', 'categorical')

feat_important_own_office = FeatureName('Important to have Own Office', 'categorical')

feat_dev_challenges = FeatureName('Devs Challenges', 'categorical')





rename_2016_dict = {'''lv0_collector''' : str(feat_collector),

                    '''lv0_country''' : str(feat_country),

                    '''lv0_un_subregion''' : str(feat_un_region),

                    '''lv0_so_region''' : str(feat_SO_region),

                    '''lv0_age_range''' : str(feat_age),

                    '''lv0_gender''' : str(feat_gender),

                    '''lv0_self_identification''' : str(feat_self_identification),

                    '''lv0_occupation''' : str(feat_occupation),

                    '''lv0_occupation_group''' : str(feat_occupation_group),

                    '''lv0_experience_range''' : str(feat_years_programming),

                    '''lv0_salary_range''' : str(feat_annual_salary_plus_bonus),

                    '''lv0_big_mac_index''' : str(feat_big_mac_index),

                    '''lv0_tech_do''' : str(feat_tech_proficiency),

                    '''lv0_tech_want''' : str(feat_tech_excitement),

                    '''lv0_aliens''' : str(feat_aliens),

                    '''lv0_programming_ability''' : str(feat_programming_ability),

                    '''lv0_employment_status''' : str(feat_employment_status),

                    '''lv0_industry''' : str(feat_industry),

                    '''lv0_company_size_range''' : str(feat_company_size),

                    '''lv0_team_size_range''' : str(feat_team_size),

                    '''lv0_women_on_team''' : str(feat_team_has_woman),

                    '''lv0_remote''' : str(feat_remote_work),

                    '''lv0_job_satisfaction''' : str(feat_job_satisfaction),

                    '''lv0_job_discovery''' : str(feat_job_discovery),

                    '''lv0_dev_environment''' : str(feat_IDE),

                    '''lv0_commit_frequency''' : str(feat_commit_frequency),

                    '''lv0_hobby''' : str(feat_program_hobby_hours_week),

                    '''lv0_dogs_vs_cats''' : str(feat_dog_cat),

                    '''lv0_desktop_os''' : str(feat_os),

                    '''lv0_unit_testing''' : str(feat_unit_testing_worth),

                    '''lv0_rep_range''' : str(feat_SO_reputation),

                    '''lv0_visit_frequency''' : str(feat_SO_freq_site),

                    '''lv0_why_learn_new_tech''' : str(feat_why_learn_new_tech),

                    '''lv0_education''' : str(feat_training_edu),

                    '''lv0_open_to_new_job''' : str(feat_job_searching),

                    '''lv0_new_job_value''' : str(feat_job_wanted_benefits),

                    '''lv0_job_search_annoyance''' : str(feat_job_search_annoyance),

                    '''lv0_interview_likelihood''' : str(feat_google_interview_likelihood),

                    '''lv0_how_to_improve_interview_process''' : str(feat_how_to_improve_interview),

                    '''lv0_star_wars_vs_star_trek''' : str(feat_star_wars_vs_star_trek),

                    '''lv0_agree_tech''' : str(feat_enjoy_tech_using),

                    '''lv0_agree_notice''' : str(feat_software_must_improve),

                    '''lv0_agree_problemsolving''' : str(feat_like_solving_problem),

                    '''lv0_agree_diversity''' : str(feat_diversity_important),

                    '''lv0_agree_adblocker''' : str(feat_use_adblock),

                    '''lv0_agree_alcohol''' : str(feat_drink_and_code),

                    '''lv0_agree_loveboss''' : str(feat_lovely_boss),

                    '''lv0_agree_nightcode''' : str(feat_love_late_night_coding),

                    '''lv0_agree_legacy''' : str(feat_legacy_code),

                    '''lv0_agree_mars''' : str(feat_wanto_go_mars),

                    '''lv0_important_variety''' : str(feat_important_variety),

                    '''lv0_important_control''' : str(feat_important_product_decision),

                    '''lv0_important_sameend''' : str(feat_important_leave_same_time),

                    '''lv0_important_newtech''' : str(feat_important_learn_tech),

                    '''lv0_important_buildnew''' : str(feat_important_build_something),

                    '''lv0_important_buildexisting''' : str(feat_important_improve_apps),

                    '''lv0_important_promotion''' : str(feat_important_promotions),

                    '''lv0_important_companymission''' : str(feat_important_company_mission),

                    '''lv0_important_wfh''' : str(feat_remote_work_relevance),

                    '''lv0_important_ownoffice''' : str(feat_important_own_office),

                    '''lv0_developer_challenges''' : str(feat_dev_challenges),

                    '''lv0_why_stack_overflow''' : str(feat_SO_why_use)}
mdf.rename_columns(2016, rename_2016_dict)
mdf.print_df_column_names(2017)
mdf.dfs[2017]['Respondent']
mdf.drop_column(2017, columns=['Respondent'])
if type(mdf.dfs[2017].columns) is pd.Index:

    new_names_2017_lv1 = mdf.get_column_names(2017)

    new_names_2017_lv0 = ["lv0_{}".format(lv1_name) for lv1_name in new_names_2017_lv1]



    mdf.set_column_multiIndex(2017, [new_names_2017_lv0, new_names_2017_lv1])

mdf.dfs[2017]
mdf.value_counts(2013, feat_occupation)
mdf.value_counts(2017, 'lv0_StackOverflowDescribes')
mdf.value_counts(2012, [feat_work_platform])
mdf.print_empty_rename_map(2017)
feat_professional_dev = FeatureName('Professional Dev', 'categorical')

feat_program_hobby = FeatureName('Program as hobby', 'categorical')

feat_enrolled_university = FeatureName('Enrolled in University Program', 'categorical')

feat_formal_edu = FeatureName('Formal Education', 'categorical')

feat_major_undergrad = FeatureName('Major Undergrad', 'categorical')

feat_company_type = FeatureName('Company Type', 'categorical')

feat_years_coding_work = FeatureName('Years Coding', 'numerical')

feat_years_coded_work = FeatureName('Years Coded', 'numerical')

feat_web_dev_type = FeatureName('Web Development Type', 'categorical')

feat_mob_dev_type = FeatureName('Mobile Developer Type', 'categorical')

feat_non_dev_type = FeatureName('Non Dev Type', 'categorical')

feat_excoder_would_code_again = FeatureName('Would Code Again', 'categorical')

feat_excoder_dev_not_enjoyable = FeatureName('Dev Not Enjoyable', 'categorical')

feat_excoder_work_life_balance = FeatureName('Better Work-Live Balance Now', 'categorical')

feat_excoder_carrer_as_planned_10yr = FeatureName('Career is as Planned', 'categorical')

feat_excoder_coder_not_belonged = FeatureName('Coder not Belonged', 'categorical')

feat_excoder_skill_out_of_date = FeatureName('Skills not up to date', 'categorical')

feat_excoder_will_not_code = FeatureName('Not Code Again', 'categorical')

feat_excode_active = FeatureName('Active Coder', 'categorical')

feat_gif_pronunciation = FeatureName('GIF Pronunciation', 'categorical')

feat_bored_by_details = FeatureName('Bored by Implementation Details', 'categorical')

feat_important_job_security = FeatureName('Important Job Security', 'categorical')

feat_annoying_poor_UI = FeatureName('Poor UI annoying', 'categorical')

feat_friends_devs = FeatureName('Friends Developers', 'categorical')

feat_wright_wrong_way = FeatureName('''There's a right and a wrong way''', 'categorical')

feat_dont_understand_computers = FeatureName('''Don't Understand Computers''', 'categorical')

feat_conscientious = FeatureName('Is Conscientious', 'categorical')

feat_invest_time_tools = FeatureName('Invest Time on Tools', 'categorical')

feat_work_pay_care = FeatureName('Work Pay Care', 'categorical')

feat_kinship = FeatureName('Kinship to Devs', 'categorical')

feat_challenge_myself = FeatureName('Challenge Myself', 'categorical')

feat_compete_peers = FeatureName('Compete With Peers', 'categorical')

feat_change_world = FeatureName('Changing the World', 'categorical')

feat_job_searching_hours = FeatureName('Job Searching Hours (week)', 'categorical')

feat_job_last_change = FeatureName('Last Changed Job', 'numerical')

feat_new_job_industry = FeatureName('New Job Industry', 'categorical')



feat_new_job_industry = FeatureName('New Job Industry', 'categorical')

feat_new_job_role = FeatureName('New Job Role', 'categorical')

feat_new_job_exp = FeatureName('New Job Exp', 'categorical')

feat_new_job_dept = FeatureName('New Job Dept', 'categorical')

feat_new_job_tech = FeatureName('New Job Tech', 'categorical')

feat_new_job_projects = FeatureName('New Job Projects', 'categorical')

feat_new_job_compensation = FeatureName('New Job Compensation', 'categorical')

feat_new_job_office_env = FeatureName('New Job Office Environment', 'categorical')

feat_new_job_commute = FeatureName('New Job Commute', 'categorical')

feat_new_job_remote = FeatureName('New Job Remote', 'categorical')

feat_new_job_leaders = FeatureName('New Job Leaders', 'categorical')

feat_new_job_prof_dev = FeatureName('New Job ProfDevel', 'categorical')

feat_new_job_diversity = FeatureName('New Job Diversity', 'categorical')

feat_new_job_product = FeatureName('New Job Product', 'categorical')

feat_new_job_finances = FeatureName('New Job Finances', 'categorical')

feat_job_benefits = FeatureName('Job Benefits', 'categorical')

feat_clicky_keys = FeatureName('Clicky Keys OK', 'categorical')

feat_job_profile_site = FeatureName('Site Job Profile', 'categorical')

feat_why_updated_CV = FeatureName('Why Updated CV', 'categorical')



feat_important_hire_algorithms = FeatureName('Important Hiring Algorithms', 'categorical')

feat_important_hire_tech_exp = FeatureName('Important Hiring TechExp', 'categorical')

feat_important_hire_communication = FeatureName('Important Hiring Communication', 'categorical')

feat_important_hire_open_source = FeatureName('Important Hiring OpenSource', 'categorical')

feat_important_hire_PM_exp = FeatureName('Important Hiring PMExp', 'categorical')

feat_important_hire_companies = FeatureName('Important Hiring Companies', 'categorical')

feat_important_hire_titles = FeatureName('Important Hiring Titles', 'categorical')

feat_important_hire_education = FeatureName('Important Hiring Education', 'categorical')

feat_important_hire_rep = FeatureName('Important Hiring Rep', 'categorical')

feat_important_hire_getting_things_done = FeatureName('Important Hiring GettingThingsDone', 'categorical')



feat_currency = FeatureName('Currency', 'categorical')

feat_overpaid = FeatureName('Overpaid', 'categorical')

feat_important_edu = FeatureName('Important Education', 'categorical')

feat_self_taught = FeatureName('Self Taught Types', 'categorical')

feat_job_wait_after_bootcamp = FeatureName('Time to Get Job after Bootcamp', 'numerical')

feat_training_advice = FeatureName('Training Advice', 'categorical')

feat_prefered_work_start = FeatureName('Prefered Time to Start Working', 'numerical')

feat_framework_proficiency = FeatureName('Framework Proficiency', 'categorical')

feat_framework_excitement = FeatureName('Framework Exciting', 'categorical')

feat_database_proficiency = FeatureName('Database Proficiency', 'categorical')

feat_database_excitement = FeatureName('Database Exciting', 'categorical')

feat_platform_proficiency = FeatureName('Platform Proficiency', 'categorical')

feat_platform_excitement = FeatureName('Platform Exciting', 'categorical')

feat_prefered_background_sound = FeatureName('Prefered Background Sound', 'categorical')

feat_methodology = FeatureName('Methodology', 'categorical')

feat_checkin_code = FeatureName('Check In Code', 'categorical')

feat_ship_now_optimize_latter = FeatureName('Ship Now Optimize Later', 'categorical')

feat_dislike_code_others = FeatureName('Deslike Code of other Devs', 'categorical')

feat_PM_techniques_useless = FeatureName('PM Techniques are Useless', 'categorical')

feat_enjoy_debugging = FeatureName('Enjoy Debugging', 'categorical')

feat_get_in_zone = FeatureName( 'Get “into the zone” when coding', 'categorical')

feat_difficult_communication = FeatureName('Difficult Communication', 'categorical')

feat_harder_collaborate_remote = FeatureName('Harder Collaborate Remote', 'categorical')

feat_best_metrics = FeatureName('Best Metrics', 'categorical')

feat_monitor_satisfaction = FeatureName('Monitor Satisfaction', 'categorical')

feat_CPU_satisfaction = FeatureName('CPU Satisfaction', 'categorical')

feat_RAM_satisfaction = FeatureName('RAM Satisfaction', 'categorical')

feat_storage_satisfaction = FeatureName('Storage Satisfaction', 'categorical')

feat_storage_IO_speed_satisfaction = FeatureName('IO Speed Satisfaction', 'categorical')



feat_influence_internet = FeatureName('Influence over Internet', 'categorical')

feat_influence_workstation = FeatureName('Influence over Workstation', 'categorical')

feat_influence_hardware = FeatureName('Influence over Hardware', 'categorical')

feat_influence_servers = FeatureName('Influence over Servers', 'categorical')

feat_influence_techStack = FeatureName('Influence over TechStack', 'categorical')

feat_influence_deptTech = FeatureName('Influence over DeptTech', 'categorical')

feat_influence_vizTools = FeatureName('Influence over VizTools', 'categorical')

feat_influence_database = FeatureName('Influence over Database', 'categorical')

feat_influence_cloud = FeatureName('Influence over Cloud', 'categorical')

feat_influence_consultants = FeatureName('Influence over Consultants', 'categorical')

feat_influence_recruitment = FeatureName('Influence over Recruitment', 'categorical')

feat_influence_communication = FeatureName('Influence over Communication', 'categorical')



feat_SO_engagement = FeatureName('StackO Engagement', 'categorical')

feat_SO_satisfaction = FeatureName('StackO Satisfaction', 'categorical')

feat_SO_device = FeatureName('Device Used to Access StackO', 'categorical')



feat_SO_recent_found_answer = FeatureName('StackO Found Answer', 'categorical')

feat_SO_recent_copied_code = FeatureName('StackO Copied Code', 'categorical')

feat_SO_recent_job_listing = FeatureName('StackO Job Listing', 'categorical')

feat_SO_recent_company_page = FeatureName('StackO Company Page', 'categorical')

feat_SO_recent_job_search = FeatureName('StackO Job Search', 'categorical')

feat_SO_recent_new_question = FeatureName('StackO New Question', 'categorical')

feat_SO_recent_answer = FeatureName('StackO Answer', 'categorical')

feat_SO_recent_meta_chat = FeatureName('StackO Meta Chat', 'categorical')



feat_ad_distraction = FeatureName('Ads Distraction', 'categorical')

feat_moderation_unfair = FeatureName('SatckO Moderation Unfair', 'categorical')

feat_SO_community_insider = FeatureName('Fell Like Member StackO', 'categorical')

feat_SO_improve_internet = FeatureName('StackO Improve Internet', 'categorical')

feat_SO_essential = FeatureName('StackO Essential', 'categorical')

feat_SO_only_money = FeatureName('StackO is All About Money', 'categorical')

feat_formal_edu_parents = feat_formal_edu + ' Parents'

feat_ethnicity = FeatureName('Ethnicity', 'categorical')

feat_survey_long = FeatureName('Survey Was Long', 'categorical')

feat_questions_interesting = FeatureName('Questions Were Interesting', 'categorical')

feat_questions_confusing = FeatureName('Questions Were Confusing', 'categorical')

feat_interested_answers = FeatureName('Interested in Answers', 'categorical')

feat_annual_salary_no_bonus = FeatureName('Annual Salary Without Bonus (USD)', 'numerical')

feat_expected_first_salary = FeatureName('Expected First Salary', 'numerical')







rename_2017_dict = {'''lv0_Professional''' : str(feat_professional_dev),

                    '''lv0_ProgramHobby''' : str(feat_program_hobby),

                    '''lv0_Country''' : str(feat_country),

                    '''lv0_University''' : str(feat_enrolled_university),

                    '''lv0_EmploymentStatus''' : str(feat_employment_status),

                    '''lv0_FormalEducation''' : str(feat_formal_edu),

                    '''lv0_MajorUndergrad''' : str(feat_major_undergrad),

                    '''lv0_HomeRemote''' : str(feat_remote_work),

                    '''lv0_CompanySize''' : str(feat_company_size),

                    '''lv0_CompanyType''' : str(feat_company_type),

                    '''lv0_YearsProgram''' : str(feat_years_programming),

                    '''lv0_YearsCodedJob''' : str(feat_years_coding_work),

                    '''lv0_YearsCodedJobPast''' : str(feat_years_coded_work),

                    '''lv0_DeveloperType''' : str(feat_occupation),

                    '''lv0_WebDeveloperType''' : str(feat_web_dev_type),

                    '''lv0_MobileDeveloperType''' : str(feat_mob_dev_type),

                    '''lv0_NonDeveloperType''' : str(feat_non_dev_type),

                    '''lv0_CareerSatisfaction''' : str(feat_career_satisfaction),

                    '''lv0_JobSatisfaction''' : str(feat_job_satisfaction),

                    '''lv0_ExCoderReturn''' : str(feat_excoder_would_code_again),

                    '''lv0_ExCoderNotForMe''' : str(feat_excoder_dev_not_enjoyable),

                    '''lv0_ExCoderBalance''' : str(feat_excoder_work_life_balance),

                    '''lv0_ExCoder10Years''' : str(feat_excoder_carrer_as_planned_10yr),

                    '''lv0_ExCoderBelonged''' : str(feat_excoder_coder_not_belonged),

                    '''lv0_ExCoderSkills''' : str(feat_excoder_skill_out_of_date),

                    '''lv0_ExCoderWillNotCode''' : str(feat_excoder_will_not_code),

                    '''lv0_ExCoderActive''' : str(feat_excode_active),

                    '''lv0_PronounceGIF''' : str(feat_gif_pronunciation),

                    '''lv0_ProblemSolving''' : str(feat_like_solving_problem),

                    '''lv0_BuildingThings''' : str(feat_important_build_something),

                    '''lv0_LearningNewTech''' : str(feat_important_learn_tech),

                    '''lv0_BoringDetails''' : str(feat_bored_by_details),

                    '''lv0_JobSecurity''' : str(feat_important_job_security),

                    '''lv0_DiversityImportant''' : str(feat_diversity_important),

                    '''lv0_AnnoyingUI''' : str(feat_annoying_poor_UI),

                    '''lv0_FriendsDevelopers''' : str(feat_friends_devs),

                    '''lv0_RightWrongWay''' : str(feat_wright_wrong_way),

                    '''lv0_UnderstandComputers''' : str(feat_dont_understand_computers),

                    '''lv0_SeriousWork''' : str(feat_conscientious),

                    '''lv0_InvestTimeTools''' : str(feat_invest_time_tools),

                    '''lv0_WorkPayCare''' : str(feat_work_pay_care),

                    '''lv0_KinshipDevelopers''' : str(feat_kinship),

                    '''lv0_ChallengeMyself''' : str(feat_challenge_myself),

                    '''lv0_CompetePeers''' : str(feat_compete_peers),

                    '''lv0_ChangeWorld''' : str(feat_change_world),

                    '''lv0_JobSeekingStatus''' : str(feat_job_searching),

                    '''lv0_HoursPerWeek''' : str(feat_job_searching_hours),

                    '''lv0_LastNewJob''' : str(feat_job_last_change),

                    '''lv0_AssessJobIndustry''' : str(feat_new_job_industry),

                    '''lv0_AssessJobRole''' : str(feat_new_job_role),

                    '''lv0_AssessJobExp''' : str(feat_new_job_exp),

                    '''lv0_AssessJobDept''' : str(feat_new_job_dept),

                    '''lv0_AssessJobTech''' : str(feat_new_job_tech),

                    '''lv0_AssessJobProjects''' : str(feat_new_job_projects),

                    '''lv0_AssessJobCompensation''' : str(feat_new_job_compensation),

                    '''lv0_AssessJobOffice''' : str(feat_new_job_office_env),

                    '''lv0_AssessJobCommute''' : str(feat_new_job_commute),

                    '''lv0_AssessJobRemote''' : str(feat_new_job_remote),

                    '''lv0_AssessJobLeaders''' : str(feat_new_job_leaders),

                    '''lv0_AssessJobProfDevel''' : str(feat_new_job_prof_dev),

                    '''lv0_AssessJobDiversity''' : str(feat_new_job_diversity),

                    '''lv0_AssessJobProduct''' : str(feat_new_job_product),

                    '''lv0_AssessJobFinances''' : str(feat_new_job_finances),

                    '''lv0_ImportantBenefits''' : str(feat_job_benefits),

                    '''lv0_ClickyKeys''' : str(feat_clicky_keys),

                    '''lv0_JobProfile''' : str(feat_job_profile_site),

                    '''lv0_ResumePrompted''' : str(feat_why_updated_CV),

                    '''lv0_LearnedHiring''' : str(feat_job_discovery),

                    '''lv0_ImportantHiringAlgorithms''' : str(feat_important_hire_algorithms),

                    '''lv0_ImportantHiringTechExp''' : str(feat_important_hire_tech_exp),

                    '''lv0_ImportantHiringCommunication''' : str(feat_important_hire_communication),

                    '''lv0_ImportantHiringOpenSource''' : str(feat_important_hire_open_source),

                    '''lv0_ImportantHiringPMExp''' : str(feat_important_hire_PM_exp),

                    '''lv0_ImportantHiringCompanies''' : str(feat_important_hire_companies),

                    '''lv0_ImportantHiringTitles''' : str(feat_important_hire_titles),

                    '''lv0_ImportantHiringEducation''' : str(feat_important_hire_education),

                    '''lv0_ImportantHiringRep''' : str(feat_important_hire_rep),

                    '''lv0_ImportantHiringGettingThingsDone''' : str(feat_important_hire_getting_things_done),

                    '''lv0_Currency''' : str(feat_currency),

                    '''lv0_Overpaid''' : str(feat_overpaid),

                    '''lv0_TabsSpaces''' : str(feat_tab_space),

                    '''lv0_EducationImportant''' : str(feat_important_edu),

                    '''lv0_EducationTypes''' : str(feat_training_edu),

                    '''lv0_SelfTaughtTypes''' : str(feat_self_taught),

                    '''lv0_TimeAfterBootcamp''' : str(feat_job_wait_after_bootcamp),

                    '''lv0_CousinEducation''' : str(feat_training_advice),

                    '''lv0_WorkStart''' : str(feat_prefered_work_start),

                    '''lv0_HaveWorkedLanguage''' : str(feat_tech_proficiency),

                    '''lv0_WantWorkLanguage''' : str(feat_tech_excitement),

                    '''lv0_HaveWorkedFramework''' : str(feat_framework_proficiency),

                    '''lv0_WantWorkFramework''' : str(feat_framework_excitement),

                    '''lv0_HaveWorkedDatabase''' : str(feat_database_proficiency),

                    '''lv0_WantWorkDatabase''' : str(feat_database_excitement),

                    '''lv0_HaveWorkedPlatform''' : str(feat_platform_proficiency),

                    '''lv0_WantWorkPlatform''' : str(feat_platform_excitement),

                    '''lv0_IDE''' : str(feat_IDE),

                    '''lv0_AuditoryEnvironment''' : str(feat_prefered_background_sound),

                    '''lv0_Methodology''' : str(feat_methodology),

                    '''lv0_VersionControl''' : str(feat_source_control_used),

                    '''lv0_CheckInCode''' : str(feat_checkin_code),

                    '''lv0_ShipIt''' : str(feat_ship_now_optimize_latter),

                    '''lv0_OtherPeoplesCode''' : str(feat_dislike_code_others),

                    '''lv0_ProjectManagement''' : str(feat_PM_techniques_useless),

                    '''lv0_EnjoyDebugging''' : str(feat_enjoy_debugging),

                    '''lv0_InTheZone''' : str(feat_get_in_zone),

                    '''lv0_DifficultCommunication''' : str(feat_difficult_communication),

                    '''lv0_CollaborateRemote''' : str(feat_harder_collaborate_remote),

                    '''lv0_MetricAssess''' : str(feat_best_metrics),

                    '''lv0_EquipmentSatisfiedMonitors''' : str(feat_monitor_satisfaction),

                    '''lv0_EquipmentSatisfiedCPU''' : str(feat_CPU_satisfaction),

                    '''lv0_EquipmentSatisfiedRAM''' : str(feat_RAM_satisfaction),

                    '''lv0_EquipmentSatisfiedStorage''' : str(feat_storage_satisfaction),

                    '''lv0_EquipmentSatisfiedRW''' : str(feat_storage_IO_speed_satisfaction),

                    '''lv0_InfluenceInternet''' : str(feat_influence_internet),

                    '''lv0_InfluenceWorkstation''' : str(feat_influence_workstation),

                    '''lv0_InfluenceHardware''' : str(feat_influence_hardware),

                    '''lv0_InfluenceServers''' : str(feat_influence_servers),

                    '''lv0_InfluenceTechStack''' : str(feat_influence_techStack),

                    '''lv0_InfluenceDeptTech''' : str(feat_influence_deptTech),

                    '''lv0_InfluenceVizTools''' : str(feat_influence_vizTools),

                    '''lv0_InfluenceDatabase''' : str(feat_influence_database),

                    '''lv0_InfluenceCloud''' : str(feat_influence_cloud),

                    '''lv0_InfluenceConsultants''' : str(feat_influence_consultants),

                    '''lv0_InfluenceRecruitment''' : str(feat_influence_recruitment),

                    '''lv0_InfluenceCommunication''' : str(feat_influence_communication),

                    '''lv0_StackOverflowDescribes''' : str(feat_SO_engagement),

                    '''lv0_StackOverflowSatisfaction''' : str(feat_SO_satisfaction),

                    '''lv0_StackOverflowDevices''' : str(feat_SO_device),

                    '''lv0_StackOverflowFoundAnswer''' : str(feat_SO_recent_found_answer),

                    '''lv0_StackOverflowCopiedCode''' : str(feat_SO_recent_copied_code),

                    '''lv0_StackOverflowJobListing''' : str(feat_SO_recent_job_listing),

                    '''lv0_StackOverflowCompanyPage''' : str(feat_SO_recent_company_page),

                    '''lv0_StackOverflowJobSearch''' : str(feat_SO_recent_job_search),

                    '''lv0_StackOverflowNewQuestion''' : str(feat_SO_recent_new_question),

                    '''lv0_StackOverflowAnswer''' : str(feat_SO_recent_answer),

                    '''lv0_StackOverflowMetaChat''' : str(feat_SO_recent_meta_chat),

                    '''lv0_StackOverflowAdsRelevant''' : str(feat_ad_rate),

                    '''lv0_StackOverflowAdsDistracting''' : str(feat_ad_distraction),

                    '''lv0_StackOverflowModeration''' : str(feat_moderation_unfair),

                    '''lv0_StackOverflowCommunity''' : str(feat_SO_community_insider),

                    '''lv0_StackOverflowHelpful''' : str(feat_SO_freq_helpful),

                    '''lv0_StackOverflowBetter''' : str(feat_SO_improve_internet),

                    '''lv0_StackOverflowWhatDo''' : str(feat_SO_essential),

                    '''lv0_StackOverflowMakeMoney''' : str(feat_SO_only_money),

                    '''lv0_Gender''' : str(feat_gender),

                    '''lv0_HighestEducationParents''' : str(feat_formal_edu_parents),

                    '''lv0_Race''' : str(feat_ethnicity),

                    '''lv0_SurveyLong''' : str(feat_survey_long),

                    '''lv0_QuestionsInteresting''' : str(feat_questions_interesting),

                    '''lv0_QuestionsConfusing''' : str(feat_questions_confusing),

                    '''lv0_InterestedAnswers''' : str(feat_interested_answers),

                    '''lv0_Salary''' : str(feat_annual_salary_no_bonus),

                    '''lv0_ExpectedSalary''' : str(feat_expected_first_salary)}
mdf.rename_columns(2017, rename_2017_dict)
mdf.print_df_column_names(2018)
drop_columns = ['Respondent']
mdf.drop_column(2018, drop_columns)
mdf.value_counts(2017, feat_program_hobby)
mdf.value_counts(2018, 'Hobby')
mdf.value_counts(2018, 'Employment')
mdf.value_counts(2017, feat_job_benefits)
if type(mdf.dfs[2018].columns) is pd.Index:

    

    new_names_2018_lv1 = mdf.get_column_names(2018)

    new_names_2018_lv0 = ["lv0_{}".format(name) for name in new_names_2018_lv1]

    

    mdf.set_column_multiIndex(2018, [new_names_2018_lv0, new_names_2018_lv1])

    

mdf.dfs[2018]
mdf.print_empty_rename_map(2018)
mdf.value_counts(2014, feat_job_contact_channel)
mdf.value_counts(2018, 'lv0_EducationTypes')
mdf.value_counts(2018, 'lv0_AdsActions')
feat_open_source = FeatureName('Open Source', 'categorical')

feat_occupation_student = FeatureName('Student', 'categorical')

feat_hope_five_years = FeatureName('Hope Five Years', 'categorical')

feat_salary = FeatureName('Salary', 'numerical')

feat_salary_period = FeatureName('Salary Period', 'categorical')

feat_currency_symbol = feat_currency + ' Symbol'

feat_communication_tools = FeatureName('Communication Tools', 'categorical')

feat_time_fully_productive = FeatureName('Time Fully Productive', 'numerical')

feat_hackathon_reasons = FeatureName('Hackathon Reasons', 'categorical')

feat_not_as_good_peers = FeatureName('Peers Are Better Coders', 'categorical')

feat_monitors_number = FeatureName('Number of Monitors', 'numerical')

feat_adblock_disabled_month = FeatureName('Disabled Adblock Last Month', 'categorical')

feat_adblock_disabled_why = FeatureName('Why Disabled Adblock', 'categorical')

feat_ad_online_valuable = FeatureName('Online Ads Valuable', 'categorical')

feat_ad_enjoy_companies = FeatureName('Like Ads Some Companies', 'categorical')

feat_ad_dislike = FeatureName('Dislike Ads', 'categorical')

feat_ad_reaction = FeatureName('Ads Reaction', 'categorical')

feat_ad_quality = FeatureName('Ads Quality', 'categorical')

feat_ia_dangerous = FeatureName('IA Dangerous', 'categorical')

feat_ia_interesting = FeatureName('IA Interesting', 'categorical')

feat_ia_responsable = FeatureName('IA Responsable', 'categorical')

feat_ia_future = FeatureName('IA Future', 'categorical')

feat_unethical_code = FeatureName('Unethical Code Doable', 'categorical')

feat_unethical_code_report = feat_unethical_code + ' report'

feat_unethical_resposible = FeatureName('Responsable for Unethical Code', 'categorical')

feat_unethical_implications = FeatureName('Unethical Implications', 'categorical')

feat_SO_recommended = FeatureName('StackO Recommended', 'categorical')

feat_SO_account = FeatureName('StackO Account', 'categorical')

feat_SO_activity_frequency = FeatureName('StackO Activity Frequency', 'numerical')

feat_SO_carrer_recommended = FeatureName('StackO Carrer Recommended', 'categorical')

feat_SO_tools_experiments = FeatureName('StackO Tools Experiments', 'categorical')

feat_wake_time_work = FeatureName('Wake Time to Work', 'numerical')

feat_hours_computer = FeatureName('Hours on Computer (day)', 'numerical')

feat_hours_outside = FeatureName('Hours Outside (day)', 'numerical')

feat_skip_meals = FeatureName('Skip Meals (week)', 'numerical')

feat_ergonomic_devices = FeatureName('Ergonomic Devices', 'categorical')

feat_exercise = FeatureName('Times Exercise (week)', 'numerical')

feat_sexual_orientation = FeatureName('Sexual Orientation', 'categorical')

feat_dependents = FeatureName('Dependents', 'numerical')

feat_military_US = FeatureName('Military US', 'categorical')

feat_survey_easy = FeatureName('Survey Easy', 'categorical')



rename_2018_dict = {'''lv0_Hobby''' : str(feat_program_hobby),

                    '''lv0_OpenSource''' : str(feat_open_source),

                    '''lv0_Country''' : str(feat_country),

                    '''lv0_Student''' : str(feat_occupation_student),

                    '''lv0_Employment''' : str(feat_employment_status),

                    '''lv0_FormalEducation''' : str(feat_formal_edu),

                    '''lv0_UndergradMajor''' : str(feat_major_undergrad),

                    '''lv0_CompanySize''' : str(feat_company_size),

                    '''lv0_DevType''' : str(feat_occupation),

                    '''lv0_YearsCoding''' : str(feat_years_programming),

                    '''lv0_YearsCodingProf''' : str(feat_years_coding_work),

                    '''lv0_JobSatisfaction''' : str(feat_job_satisfaction),

                    '''lv0_CareerSatisfaction''' : str(feat_career_satisfaction),

                    '''lv0_HopeFiveYears''' : str(feat_hope_five_years),

                    '''lv0_JobSearchStatus''' : str(feat_job_searching),

                    '''lv0_LastNewJob''' : str(feat_job_last_change),

                    '''lv0_AssessJob1''' : str(feat_new_job_industry),

                    '''lv0_AssessJob2''' : str(feat_new_job_finances),

                    '''lv0_AssessJob3''' : str(feat_new_job_dept),

                    '''lv0_AssessJob4''' : str(feat_new_job_tech),

                    '''lv0_AssessJob5''' : str(feat_new_job_compensation),

                    '''lv0_AssessJob6''' : str(feat_new_job_office_env),

                    '''lv0_AssessJob7''' : str(feat_new_job_remote),

                    '''lv0_AssessJob8''' : str(feat_new_job_prof_dev),

                    '''lv0_AssessJob9''' : str(feat_new_job_dept),

                    '''lv0_AssessJob10''' : str(feat_new_job_product),

                    '''lv0_AssessBenefits1''' : str(feat_job_benefits),

                    '''lv0_AssessBenefits2''' : str(feat_job_benefits),

                    '''lv0_AssessBenefits3''' : str(feat_job_benefits),

                    '''lv0_AssessBenefits4''' : str(feat_job_benefits),

                    '''lv0_AssessBenefits5''' : str(feat_job_benefits),

                    '''lv0_AssessBenefits6''' : str(feat_job_benefits),

                    '''lv0_AssessBenefits7''' : str(feat_job_benefits),

                    '''lv0_AssessBenefits8''' : str(feat_job_benefits),

                    '''lv0_AssessBenefits9''' : str(feat_job_benefits),

                    '''lv0_AssessBenefits10''' : str(feat_job_benefits),

                    '''lv0_AssessBenefits11''' : str(feat_job_benefits),

                    '''lv0_JobContactPriorities1''' : str(feat_job_contact_channel),

                    '''lv0_JobContactPriorities2''' : str(feat_job_contact_channel),

                    '''lv0_JobContactPriorities3''' : str(feat_job_contact_channel),

                    '''lv0_JobContactPriorities4''' : str(feat_job_contact_channel),

                    '''lv0_JobContactPriorities5''' : str(feat_job_contact_channel),

                    '''lv0_JobEmailPriorities1''' : str(feat_job_msg_appealing),

                    '''lv0_JobEmailPriorities2''' : str(feat_job_msg_appealing),

                    '''lv0_JobEmailPriorities3''' : str(feat_job_msg_appealing),

                    '''lv0_JobEmailPriorities4''' : str(feat_job_msg_appealing),

                    '''lv0_JobEmailPriorities5''' : str(feat_job_msg_appealing),

                    '''lv0_JobEmailPriorities6''' : str(feat_job_msg_appealing),

                    '''lv0_JobEmailPriorities7''' : str(feat_job_msg_appealing),

                    '''lv0_UpdateCV''' : str(feat_why_updated_CV),

                    '''lv0_Currency''' : str(feat_currency),

                    '''lv0_Salary''' : str(feat_salary),

                    '''lv0_SalaryType''' : str(feat_salary_period),

                    '''lv0_ConvertedSalary''' : str(feat_annual_salary_no_bonus),# It was not explicitly mentioned to include bonus

                    '''lv0_CurrencySymbol''' : str(feat_currency_symbol),

                    '''lv0_CommunicationTools''' : str(feat_communication_tools),

                    '''lv0_TimeFullyProductive''' : str(feat_time_fully_productive),

                    '''lv0_EducationTypes''' : str(feat_training_edu),

                    '''lv0_SelfTaughtTypes''' : str(feat_self_taught),

                    '''lv0_TimeAfterBootcamp''' : str(feat_job_wait_after_bootcamp),

                    '''lv0_HackathonReasons''' : str(feat_hackathon_reasons),

                    '''lv0_AgreeDisagree1''' : str(feat_kinship),

                    '''lv0_AgreeDisagree2''' : str(feat_compete_peers),

                    '''lv0_AgreeDisagree3''' : str(feat_not_as_good_peers),

                    '''lv0_LanguageWorkedWith''' : str(feat_tech_proficiency),

                    '''lv0_LanguageDesireNextYear''' : str(feat_tech_excitement),

                    '''lv0_DatabaseWorkedWith''' : str(feat_database_proficiency),

                    '''lv0_DatabaseDesireNextYear''' : str(feat_database_excitement),

                    '''lv0_PlatformWorkedWith''' : str(feat_platform_proficiency),

                    '''lv0_PlatformDesireNextYear''' : str(feat_platform_excitement),

                    '''lv0_FrameworkWorkedWith''' : str(feat_framework_proficiency),

                    '''lv0_FrameworkDesireNextYear''' : str(feat_framework_excitement),

                    '''lv0_IDE''' : str(feat_IDE),

                    '''lv0_OperatingSystem''' : str(feat_os),

                    '''lv0_NumberMonitors''' : str(feat_monitors_number),

                    '''lv0_Methodology''' : str(feat_methodology),

                    '''lv0_VersionControl''' : str(feat_source_control_used),

                    '''lv0_CheckInCode''' : str(feat_checkin_code),

                    '''lv0_AdBlocker''' : str(feat_use_adblock),

                    '''lv0_AdBlockerDisable''' : str(feat_adblock_disabled_month),

                    '''lv0_AdBlockerReasons''' : str(feat_adblock_disabled_why),

                    '''lv0_AdsAgreeDisagree1''' : str(feat_ad_online_valuable),

                    '''lv0_AdsAgreeDisagree2''' : str(feat_ad_enjoy_companies),

                    '''lv0_AdsAgreeDisagree3''' : str(feat_ad_dislike),

                    '''lv0_AdsActions''' : str(feat_ad_reaction),

                    '''lv0_AdsPriorities1''' : str(feat_ad_quality),

                    '''lv0_AdsPriorities2''' : str(feat_ad_quality),

                    '''lv0_AdsPriorities3''' : str(feat_ad_quality),

                    '''lv0_AdsPriorities4''' : str(feat_ad_quality),

                    '''lv0_AdsPriorities5''' : str(feat_ad_quality),

                    '''lv0_AdsPriorities6''' : str(feat_ad_quality),

                    '''lv0_AdsPriorities7''' : str(feat_ad_quality),

                    '''lv0_AIDangerous''' : str(feat_ia_dangerous),

                    '''lv0_AIInteresting''' : str(feat_ia_interesting),

                    '''lv0_AIResponsible''' : str(feat_ia_responsable),

                    '''lv0_AIFuture''' : str(feat_ia_future),

                    '''lv0_EthicsChoice''' : str(feat_unethical_code),

                    '''lv0_EthicsReport''' : str(feat_unethical_code_report),

                    '''lv0_EthicsResponsible''' : str(feat_unethical_resposible),

                    '''lv0_EthicalImplications''' : str(feat_unethical_implications),

                    '''lv0_StackOverflowRecommend''' : str(feat_SO_recommended),

                    '''lv0_StackOverflowVisit''' : str(feat_SO_freq_site),

                    '''lv0_StackOverflowHasAccount''' : str(feat_SO_account),

                    '''lv0_StackOverflowParticipate''' : str(feat_SO_activity_frequency),

                    '''lv0_StackOverflowJobs''' : str(feat_SO_carrer_aware),

                    '''lv0_StackOverflowDevStory''' : str(feat_SO_carrer_profile),

                    '''lv0_StackOverflowJobsRecommend''' : str(feat_SO_carrer_recommended),

                    '''lv0_StackOverflowConsiderMember''' : str(feat_SO_community_insider),

                    '''lv0_HypotheticalTools1''' : str(feat_SO_tools_experiments),

                    '''lv0_HypotheticalTools2''' : str(feat_SO_tools_experiments),

                    '''lv0_HypotheticalTools3''' : str(feat_SO_tools_experiments),

                    '''lv0_HypotheticalTools4''' : str(feat_SO_tools_experiments),

                    '''lv0_HypotheticalTools5''' : str(feat_SO_tools_experiments),

                    '''lv0_WakeTime''' : str(feat_wake_time_work),

                    '''lv0_HoursComputer''' : str(feat_hours_computer),

                    '''lv0_HoursOutside''' : str(feat_hours_outside),

                    '''lv0_SkipMeals''' : str(feat_skip_meals),

                    '''lv0_ErgonomicDevices''' : str(feat_ergonomic_devices),

                    '''lv0_Exercise''' : str(feat_exercise),

                    '''lv0_Gender''' : str(feat_gender),

                    '''lv0_SexualOrientation''' : str(feat_sexual_orientation),

                    '''lv0_EducationParents''' : str(feat_formal_edu_parents),

                    '''lv0_RaceEthnicity''' : str(feat_ethnicity),

                    '''lv0_Age''' : str(feat_age),

                    '''lv0_Dependents''' : str(feat_dependents),

                    '''lv0_MilitaryUS''' : str(feat_military_US),

                    '''lv0_SurveyTooLong''' : str(feat_survey_long),

                    '''lv0_SurveyEasy''' : str(feat_survey_easy)}
mdf.rename_columns(2018, rename_2018_dict)
mdf.print_df_column_names(2019)
drop_columns = ['Respondent']
mdf.drop_column(2019, drop_columns)
if type(mdf.dfs[2019].columns) is pd.Index:

    lv1_names_2019 = mdf.get_column_names(2019)

    lv0_names_2019 = ["lv0_{}".format(n) for n in lv1_names_2019]

    mdf.set_column_multiIndex(2019, [lv0_names_2019, lv1_names_2019])

mdf.dfs[2019]
mdf.print_empty_rename_map(2019)
mdf.value_counts(2017, feat_professional_dev)
mdf.value_counts(2016, feat_employment_status)
mdf.value_counts(2018, feat_occupation_student)
mdf.value_counts(2012, feat_occupation)
mdf.value_counts(2019, 'lv0_DevEnviron')
feat_open_source_quality = feat_open_source + ' Quality'

feat_age_first_code = FeatureName('Age First Code', 'numerical')

feat_manager_competent = FeatureName('Manager Competent', 'categorical')

feat_manager_money = FeatureName('Manager Make Money', 'categorical')

feat_manager_want = FeatureName('Want Become Manager', 'categorical')

feat_interview_test = FeatureName('Last Successful Interview Test', 'numerical')

feat_fizz_buzz = FeatureName('Solved Fizz Buzz', 'categorical')

feat_when_updated_CV = FeatureName( 'When Updated CV', 'categorical')

feat_salary_plus_bonus = FeatureName('Salary (+bonus)', 'numerical')

feat_work_week_hours = FeatureName('Work Week Hours', 'numerical')

feat_planned_work = FeatureName('Work Planned', 'categorical')

feat_productivity_challenge = FeatureName('Productivity Challenge', 'categorical')

feat_work_location = FeatureName('Work Location', 'categorical')

feat_self_assessment = FeatureName('Self Assessment', 'categorical')

feat_code_reviewer = FeatureName('Code Reviewer', 'categorical')

feat_code_reviewing = FeatureName('Time Reviewing (hrs)', 'numerical')

feat_unit_test_company = FeatureName('Company Apply Unit Tests', 'categorical')

feat_company_purchase_how = FeatureName('Company Purchase Decision Maker', 'categorical')

feat_tech_misc_proficiency = FeatureName('Technology Miscellaneous', 'categorical')

feat_tech_misc_excitement = FeatureName('Exciting Technology Miscellaneous', 'categorical')

feat_containers = FeatureName('Using Containers', 'categorical')

feat_blockchain = FeatureName('Using Blockchain', 'categorical')

feat_blockchain_perception = FeatureName('Blockchain Perception', 'categorical')

feat_better_life = FeatureName('Next Generation Better Life', 'categorical')

feat_it_person = FeatureName('Is the IT Person in Family', 'categorical')

feat_off_on = FeatureName('Tried Turning off and on', 'categorical')

feat_social_media = FeatureName('Used Social Media', 'categorical')

feat_online_or_IRL = FeatureName('Online or Real Life Conversation', 'categorical')

feat_screen_name = FeatureName('Screen Name', 'categorical')

feat_SO_first_visit = FeatureName('StackO First Visit', 'categorical')

feat_SO_more_time_saved = FeatureName('StackO Save More Time', 'categorical')

feat_SO_time_saved = FeatureName('StackO Time Saved', 'numerical')

feat_SO_carrer_visit = FeatureName('StackO Career Visit', 'categorical')

feat_SO_enterprise = FeatureName('StackO Enterprise', 'categorical')

feat_SO_change = FeatureName('Feel More Welcome this Year', 'categorical')

feat_SO_new_content = FeatureName('Wanted New Content', 'categorical')

feat_trans = FeatureName('Trans', 'categorical')



rename_2019_dict = {'''lv0_MainBranch''' : str(feat_professional_dev),

                    '''lv0_Hobbyist''' : str(feat_program_hobby),

                    '''lv0_OpenSourcer''' : str(feat_open_source),

                    '''lv0_OpenSource''' : str(feat_open_source_quality),

                    '''lv0_Employment''' : str(feat_employment_status),

                    '''lv0_Country''' : str(feat_country),

                    '''lv0_Student''' : str(feat_occupation_student),

                    '''lv0_EdLevel''' : str(feat_formal_edu),

                    '''lv0_UndergradMajor''' : str(feat_major_undergrad),

                    '''lv0_EduOther''' : str(feat_self_taught),

                    '''lv0_OrgSize''' : str(feat_company_size),

                    '''lv0_DevType''' : str(feat_occupation),

                    '''lv0_YearsCode''' : str(feat_years_programming),

                    '''lv0_Age1stCode''' : str(feat_age_first_code),

                    '''lv0_YearsCodePro''' : str(feat_years_coding_work),

                    '''lv0_CareerSat''' : str(feat_career_satisfaction),

                    '''lv0_JobSat''' : str(feat_job_satisfaction),

                    '''lv0_MgrIdiot''' : str(feat_manager_competent),

                    '''lv0_MgrMoney''' : str(feat_manager_money),

                    '''lv0_MgrWant''' : str(feat_manager_want),

                    '''lv0_JobSeek''' : str(feat_job_searching),

                    '''lv0_LastHireDate''' : str(feat_job_last_change),

                    '''lv0_LastInt''' : str(feat_interview_test),

                    '''lv0_FizzBuzz''' : str(feat_fizz_buzz),

                    '''lv0_JobFactors''' : str(feat_job_wanted_benefits),

                    '''lv0_ResumeUpdate''' : str(feat_when_updated_CV),

                    '''lv0_CurrencySymbol''' : str(feat_currency_symbol),

                    '''lv0_CurrencyDesc''' : str(feat_currency),

                    '''lv0_CompTotal''' : str(feat_salary_plus_bonus),

                    '''lv0_CompFreq''' : str(feat_salary_period),

                    '''lv0_ConvertedComp''' : str(feat_annual_salary_plus_bonus),

                    '''lv0_WorkWeekHrs''' : str(feat_work_week_hours),

                    '''lv0_WorkPlan''' : str(feat_planned_work),

                    '''lv0_WorkChallenge''' : str(feat_productivity_challenge),

                    '''lv0_WorkRemote''' : str(feat_remote_work),

                    '''lv0_WorkLoc''' : str(feat_work_location),

                    '''lv0_ImpSyn''' : str(feat_self_assessment),

                    '''lv0_CodeRev''' : str(feat_code_reviewer),

                    '''lv0_CodeRevHrs''' : str(feat_code_reviewing),

                    '''lv0_UnitTests''' : str(feat_unit_test_company),

                    '''lv0_PurchaseHow''' : str(feat_company_purchase_how),

                    '''lv0_PurchaseWhat''' : str(feat_purch_role),

                    '''lv0_LanguageWorkedWith''' : str(feat_tech_proficiency),

                    '''lv0_LanguageDesireNextYear''' : str(feat_tech_excitement),

                    '''lv0_DatabaseWorkedWith''' : str(feat_database_proficiency),

                    '''lv0_DatabaseDesireNextYear''' : str(feat_database_excitement),

                    '''lv0_PlatformWorkedWith''' : str(feat_platform_proficiency),

                    '''lv0_PlatformDesireNextYear''' : str(feat_platform_excitement),

                    '''lv0_WebFrameWorkedWith''' : str(feat_framework_proficiency),

                    '''lv0_WebFrameDesireNextYear''' : str(feat_framework_excitement),

                    '''lv0_MiscTechWorkedWith''' : str(feat_tech_misc_proficiency),

                    '''lv0_MiscTechDesireNextYear''' : str(feat_tech_misc_excitement),

                    '''lv0_DevEnviron''' : str(feat_IDE),

                    '''lv0_OpSys''' : str(feat_os),

                    '''lv0_Containers''' : str(feat_containers),

                    '''lv0_BlockchainOrg''' : str(feat_blockchain),

                    '''lv0_BlockchainIs''' : str(feat_blockchain_perception),

                    '''lv0_BetterLife''' : str(feat_better_life),

                    '''lv0_ITperson''' : str(feat_it_person),

                    '''lv0_OffOn''' : str(feat_off_on),

                    '''lv0_SocialMedia''' : str(feat_social_media),

                    '''lv0_Extraversion''' : str(feat_online_or_IRL),

                    '''lv0_ScreenName''' : str(feat_screen_name),

                    '''lv0_SOVisit1st''' : str(feat_SO_first_visit),

                    '''lv0_SOVisitFreq''' : str(feat_SO_freq_site),

                    '''lv0_SOVisitTo''' : str(feat_SO_why_use),

                    '''lv0_SOFindAnswer''' : str(feat_SO_recent_answer),

                    '''lv0_SOTimeSaved''' : str(feat_SO_more_time_saved),

                    '''lv0_SOHowMuchTime''' : str(feat_SO_time_saved),

                    '''lv0_SOAccount''' : str(feat_SO_account),

                    '''lv0_SOPartFreq''' : str(feat_SO_activity_frequency),

                    '''lv0_SOJobs''' : str(feat_SO_carrer_visit),

                    '''lv0_EntTeams''' : str(feat_SO_enterprise),

                    '''lv0_SOComm''' : str(feat_SO_community_insider),

                    '''lv0_WelcomeChange''' : str(feat_SO_change),

                    '''lv0_SONewContent''' : str(feat_SO_new_content),

                    '''lv0_Age''' : str(feat_age),

                    '''lv0_Gender''' : str(feat_gender),

                    '''lv0_Trans''' : str(feat_trans),

                    '''lv0_Sexuality''' : str(feat_sexual_orientation),

                    '''lv0_Ethnicity''' : str(feat_ethnicity),

                    '''lv0_Dependents''' : str(feat_dependents),

                    '''lv0_SurveyLength''' : str(feat_survey_long),

                    '''lv0_SurveyEase''' : str(feat_survey_easy)}
mdf.rename_columns(2019, rename_2019_dict)
mdf.show_shared_columns([2011, 2017])
shared_columns = mdf.show_shared_columns()
shared_columns.shape
shared_columns.head(60)
common_columns = mdf.find_common_columns()
common_columns.head(20)
mdf.append()
uv = mdf.unique_values()
uv.count().sort_values(ascending=True).head(30)
uv.count().sort_values(ascending=False).head(30)
mdf.appended[("Exciting Tech","LanguageDesireNextYear")].unique()