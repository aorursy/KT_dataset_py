### Create user class

class User:
    handles = []
    def __init__(self, full_name, handle):
        self._full_name = full_name
        self._handle = handle
        self._friends = []
        self._followers = []
        try:
            if (handle in self.handles):
                raise ValueError("Username already exists")
        except ValueError as e: 
            print(e)
        self.handles.append(handle)
    def add_follower(self, follower): 
        self._followers.append(follower)
    def add_friend(self, friend): 
        self._friends.append(friend)
    def remove_follower(self, follower): 
        self._followers.remove(follower) 
    def remove_friend(self, friend): 
        self._friends.remove(friend)
    def has_follower(self, follower):
        return follower in self._followers
    def has_friend(self, friend):
        return friend in self._friends
    def get_full_name(self):
        return self._full_name
    def set_full_name(self, full_name):
        self._full_name = full_name
    def get_handle(self):
        return self._handle
    def set_handle(self, handle):
        self._handle = handle
    def get_friends(self):
        return self._friends
    def set_friends(self, friends):
        self._friends = friends
    def get_followers(self):
        return self._followers
    def set_followers(self, followers):
        self._followers = followers
    def __str__(self):
        return """User name: {}\nHandle: {}\nFriends: {}\nFollowers: {}\n""".format(self._full_name, self._handle, [friend.get_full_name() for friend in self._friends], [follower.get_full_name() for follower in self._followers])
### Create network class

class Network:
    def __init__(self):
        self._users = []
    def add_user(self, full_name, handle):
        try: 
            if (self._check_handle(handle)): 
                user = User(full_name, handle)
                self._users.append(user)
        except ValueError as e: 
            print(e)
    
    def set_relation(self, user_1, user_2, rel_type): 
        
        ''' sets relations between users and checks if users are already friends
        or followers and raises error if so '''
        
        try: 
            if (not type(user_1) == User or not type(user_2 == User)):
                raise TypeError("Incorrect types, excepted User type")
            if (self._handle_exists(user_1.get_handle()) and self._handle_exists(user_2.get_handle())):
                if rel_type == "follower":
                    if (user_2 in user_1.get_followers() or user_2 in user_1.get_friends()):
                        raise ValueError("Relationship is not allowed")
                    user_1.add_follower(user_2)
                elif rel_type == "friend":
                    if (user_2 in user_1.get_friends()):
                        raise ValueError("You are already friends")
                    if(user_2 in user_1.get_followers()):
                        user_1.remove_follower(user_2)
                    if(user_1 in user_2.get_followers()):
                        user_2.remove_follower(user_1)
                    user_1.add_friend(user_2) # adds friend
                    user_2.add_friend(user_1) # adds friend back and relation between user_1 and _2.
        except (ValueError, TypeError) as e:
            print(e)
    
    def block(self, user_1, user_2):
        
        ''' block users: 
        raise error if user_1 and,or user_2 don't share a relation '''
        
        try: 
            if (not type(user_1) == User or not type(user_2 == User)):
                raise TypeError("Incorrect types, excepted User type")
            if (self._have_relation(user_1, user_2)):
                if (user_1.has_follower(user_2)):
                    user_1.remove_follower(user_2)
                if (user_2.has_follower(user_1)): 
                    user_2.remove_follower(user_1)
                if (user_1.has_friend(user_2)):
                    user_1.remove_friend(user_2)
                    user_2.remove_friend(user_1)
        except (ValueError, TypeError) as e: 
            print(e)
    
    def _have_relation(self, user_1, user_2): 
        if (user_1 in user_2.get_followers() or 
            user_2 in user_1.get_followers() or 
            user_1 in user_2.get_friends()):
            return True
        raise ValueError("User {} and user {} do not have a relationship".format(user_1, user_2))                      
    def _check_handle(self, handle): 
        if (handle.isalnum() and 
           handle not in User.handles):
            return True
        raise ValueError("Handle: {} is not suitable".format(handle)) 
    def _handle_exists(self, handle):
        if handle in User.handles:
            return True 
        raise ValueError ("Handle: {} is non-existent".format(handle))    
    def _handle_is_alphanumeric(self, handle):
        if handle.isalnum():
            return True 
        raise ValueError ("Handle: {} is not alphanumeric".format(handle)) 
    def __str__(self): 
        return "Users of this network are: \n{}".format("\n".join([str(user) for user in self._users]))
network = Network()
network.add_user("Simon", "simon123")
network.add_user("Symrun", "symrun123")
network.add_user("Giulia", "giulia123")
user_1 = network._users[0]
user_2 = network._users[1]

network.set_relation(user_1, user_2, "friend")

print(network)