class Solution(object):
    def numDecodings(self,s):
        if not s:
            return 0
        if s[0] == '0':
            return 0
        n = len(s)
        count1 = count2 = 1
        for i in range(1,n):
            if s[i] == 0:
                if s[i-1] not in'12':
                    return 0
                else:
                    count = count1
            else:
                if 10 <= int(s[i-1:i+1]) <= 26:
                    count = count1+count2
                else:
                    count = count2
            count1,count2 = count2,count1 
            
        return count2 
ob1 = Solution()
print(ob1.numDecodings("122"))    
                
