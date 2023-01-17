import time
import pandas as pd
import numpy as np

CITY_DATA = { 'chicago': 'chicago.csv',
              'new york': 'new_york_city.csv',
              'washington': 'washington.csv' }

def get_filters():
    """
    Asks user to specify a city, month, and day to analyze.

    Returns:
        (str) city - name of the city to analyze
        (str) month - name of the month to filter by, or "all" to apply no month filter
        (str) day - name of the day of week to filter by, or "all" to apply no day filter
    """
    print('Hello! Let\'s explore some US bikeshare data!')
    # TO DO: get user input for city (chicago, new york city, washington). HINT: Use a while loop to handle invalid inputs
    while True:
        city=input('\nWould you like to see data for Chicago, New York, or Washington?\n')
        city=city.lower()
        if city not in ['chicago','new york','washington','ch','ny','wa']:
            print('Please choose between Chicago, New York , or Washington (\n')
        else:
             print('\nYou chose {}! We\'re going to explore its bikeshare data\n'.format(city))
             break
    # TO DO: get user input for month (all, january, february, ... , june)
    while True:
        month=input('Please enter the month you want the data filtered from as(all,january,february...,june) :  ')
        month=month.lower()
        if month not in ['all','january','february','march','april','may','june']:
            print('You have not entered the month as either January, February, March, April, May,June or all.Please reneter\n')
        else:
            break
        
    # TO DO: get user input for day of week (all, monday, tuesday, ... sunday)
    while True:
        day=input('Please enter the week you want the data filtered from as(all,monday,tuesday,....,sunday): ')
        day=day.lower()
        if day not in ['all','monday','tuesday','wednesday','thursday','friday','saturday','sunday']:
            print('You have not entered the data as eihter all,monday,tuesday,wednesday,Thursday,friday, saturday or sunday.Please renter\n')
        else:
            break


    print('-'*100)
    return city, month, day 


def load_data(city, month, day):
    """
    Loads data for the specified city and filters by month and day if applicable.

    Args:
        (str) city - name of the city to analyze
        (str) month - name of the month to filter by, or "all" to apply no month filter
        (str) day - name of the day of week to filter by, or "all" to apply no day filter
    Returns:
        df - Pandas DataFrame containing city data filtered by month and day
    """
    # load data file into a dataframe
    df = pd.read_csv(CITY_DATA[city])
    
    # convert the Start Time column to datetime
    df['Start Time'] = pd.to_datetime(df['Start Time'])
    
    # extract month and day of week from Start Time to create new columns
    df['month'] = df['Start Time'].dt.month
    df['day_of_week'] = df['Start Time'].dt.weekday_name
    # extract hour from the Start Time column to create an hour column
    df['hour'] = df['Start Time'].dt.hour
    # filter by month if applicable
    if month != 'all':
        
        # use the index of the months list to get the corresponding int
        months = ['january', 'february', 'march', 'april', 'may', 'june']
        month = months.index(month) + 1
        # filter by month to create the new dataframe
        df = df[df['month']==month]
        
      # filter by day of week if applicable
    if day != 'all':
        # filter by day of week to create the new dataframe
        df = df = df[df['day_of_week'] == day.title()]
        
    return df

def display_raw_data(df):
    """
    Displays the data used to compute the stats
    Input:
        the dataframe with all the bikeshare data
    Returns: 
       none
    """
    #remove the row header
    df = df.drop(['month'], axis = 1)
    
    row_index = 0
    seeData = input("\n Would you like to see rows of the data used to compute the stats? Please write 'yes' or 'no' \n").lower()
    while True:

        if seeData == 'no':
            return

        if seeData == 'yes':
            print(df[rowindex: row_index + 5])
            row_index = row_index + 5

        
        seeData = input("\n Would you like to see five more rows of the data used to compute the stats? Please write 'yes' or 'no' \n").lower()




def time_stats(df):
    """Displays statistics on the most frequent times of travel."""

    print('\nCalculating The Most Frequent Times of Travel...\n')
    start_time = time.time()

    # TO DO: display the most common month
    popular_month = df['month'].mode()[0]
    print('The most common month of travel is: {}\n'.format(popular_month))

    # TO DO: display the most common day of week
    popular_day_of_week = df['day_of_week'].mode()[0]
    print('The most common day of travel is : {}\n'.format(popular_day_of_week))

    # TO DO: display the most common start hour
    popular_hour = df['hour'].mode()[0]
    print('The most common start hour is : {}\n'.format(popular_hour))


    print("\nThis took %s seconds." % (time.time() - start_time))
    print('-'*100)


def station_stats(df):
    """Displays statistics on the most popular stations and trip."""

    print('\nCalculating The Most Popular Stations and Trip...\n')
    start_time = time.time()

    # TO DO: display most commonly used start station
    popular_start_station = df['Start Station'].mode()[0]
    print('The most common Start Station is : {}\n'.format(popular_start_station))

    # TO DO: display most commonly used end station
    popular_end_station = df['End Station'].mode()[0]
    print('The most common End Station is : {}\n'.format(popular_end_station))

    # TO DO: display most frequent combination of start station and end station trip
    popular_start_and_end_station = df.groupby(['Start Station','End Station']).size().nlargest(1)

    print("\nThis took %s seconds." % (time.time() - start_time))
    print('-'*100)


def trip_duration_stats(df):
    """Displays statistics on the total and average trip duration."""

    print('\nCalculating Trip Duration...\n')
    start_time = time.time()
    #converting end time to datetime 
    df['End Time'] = pd.to_datetime(df['End Time'])

    # TO DO: display total travel time
    df['Travel Time'] = df['End Time'] - df['Start Time']
    
    #sum for total trip time, mean for avg trip time
    total_travel_time = np.sum(df['Travel Time'])

    # TO DO: display mean travel time
    average_travel_time = np.mean(df['Travel Time'])

    print("\nThis took %s seconds." % (time.time() - start_time))
    print('-'*100)


def user_stats(df):
    """Displays statistics on bikeshare users."""

    print('\nCalculating User Stats...\n')
    start_time = time.time()

    # TO DO: Display counts of user types
    user_types = df['User Type'].value_counts()

    # TO DO: Display counts of gender
    try:
        gender = df['Gender'].value_counts()
    except:
        print('Gender is not available for this city.')

    # TO DO: Display earliest, most recent, and most common year of birth
    try:
        earliest = np.min(df['Birth Year'])
        print ("\nThe earliest year of birth is {}\n".format(earliest))
        most_recent = np.max(df['Birth Year'])
        print ("The latest year of birth is {}\n".format(most_recent))
        most_common= df['Birth Year'].mode()[0]
        print ("The most frequent year of birth is {}\n".format(most_common))
    except:
        print('Birth date is not available for this city.')


    print("\nThis took %s seconds." % (time.time() - start_time))
    print('-'*100)


def main():
    while True:
        city, month, day = get_filters()
        df = load_data(city, month, day)
        display_raw_data(df)     

        time_stats(df)
        station_stats(df)
        trip_duration_stats(df)
        user_stats(df)

        restart = input('\nWould you like to restart? Enter yes or no.\n')
        if restart.lower() != 'yes':
            break


if __name__ == "__main__":
	main()